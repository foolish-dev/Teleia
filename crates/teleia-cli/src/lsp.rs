//! Minimal LSP (Language Server Protocol) client. Spawns each
//! configured server, exchanges the `initialize` handshake over the
//! Content-Length-framed JSON-RPC the protocol mandates, and exposes
//! `textDocument/diagnostic` (pull diagnostics, LSP 3.17) and
//! `textDocument/hover`, `textDocument/definition`,
//! `textDocument/references` and `workspace/symbol` as the
//! `lsp_diagnostics`, `lsp_hover`, `lsp_definition`, `lsp_references`
//! and `lsp_symbols` agent tools.
//!
//! Document synchronisation is best-effort: a file gets a one-shot
//! `textDocument/didOpen` the first time the agent asks about it, then
//! requests fan out to every running server so we don't have to
//! maintain a language-id → server routing table here.
//!
//! Positions cross the agent boundary 1-based, matching how every
//! `lsp_*` row prints them, and are converted to LSP's 0-based wire
//! form exactly once per direction — so any printed row feeds straight
//! back into any position-taking tool. Columns are the server's own,
//! which the spec measures in UTF-16 code units: invisible while
//! positions round-trip through our own rows, wrong for a column
//! counted out of `grep` output on a line with non-ASCII before it.
//! Closing that gap needs encoding negotiation on both boundaries and
//! is not done here.

use anyhow::{anyhow, Context, Result};
use futures_util::future::BoxFuture;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use teleia_agent::ToolRouter;
use teleia_llm::ToolDef;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout, Command};

use crate::config::LspEntry;

/// Synthetic tool name surfaced by [`LspRegistry`] when at least one
/// LSP server is running. Pulls diagnostics for a file via
/// `textDocument/diagnostic`.
pub const DIAGNOSTICS_TOOL: &str = "lsp_diagnostics";

/// Synthetic tool name surfaced by [`LspRegistry`] when at least one
/// LSP server is running. Requests hover info for a 1-based
/// `(line, character)` position via `textDocument/hover`.
pub const HOVER_TOOL: &str = "lsp_hover";

/// Synthetic tool name surfaced by [`LspRegistry`] when at least one
/// LSP server is running. Resolves the symbol at a 1-based
/// `(line, character)` to where it is defined, via
/// `textDocument/definition`.
pub const DEFINITION_TOOL: &str = "lsp_definition";

/// Synthetic tool name surfaced by [`LspRegistry`] when at least one
/// LSP server is running. Lists every use of the symbol at a 1-based
/// `(line, character)` via `textDocument/references`.
pub const REFERENCES_TOOL: &str = "lsp_references";

/// Synthetic tool name surfaced by [`LspRegistry`] when at least one
/// LSP server is running. Searches the workspace symbol index by name
/// via `workspace/symbol`. Unlike the others it takes no `path`: the
/// query goes to the server's index, not to a document.
pub const SYMBOLS_TOOL: &str = "lsp_symbols";

/// Every tool [`LspRegistry`] answers, driving both `definitions` and
/// `handles` off one list. A name in one but not the other is not a
/// local error: `Agent::is_routed` would hand the call to the built-in
/// dispatcher, which fails it as an unknown tool the model can see in
/// its own catalogue — and then retries.
pub const LSP_TOOLS: &[&str] = &[
    DIAGNOSTICS_TOOL,
    HOVER_TOOL,
    DEFINITION_TOOL,
    REFERENCES_TOOL,
    SYMBOLS_TOOL,
];

#[derive(Debug, Clone, Deserialize)]
struct Position {
    line: u32,
    character: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct Range {
    start: Position,
    #[allow(dead_code)]
    end: Position,
}

#[derive(Debug, Clone, Deserialize)]
struct Diagnostic {
    range: Range,
    #[serde(default)]
    severity: Option<u8>,
    #[serde(default)]
    message: String,
    #[serde(default)]
    source: Option<String>,
}

pub struct LspClient {
    pub name: String,
    pub server_name: Option<String>,
    pub server_version: Option<String>,
    child: tokio::process::Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    /// URIs the agent has pushed a `didOpen` for. A re-query bumps the
    /// version and pushes a `didChange` with the current on-disk text, so
    /// the server never keeps reporting against the stale buffer it first
    /// opened. The recorded mtime is what lets
    /// [`LspClient::resync_open_documents`] refresh the *other* open
    /// documents before a request whose answer can name them.
    open_docs: HashMap<String, OpenDoc>,
    /// Set by a request that timed out. The timeout fires mid-frame, so
    /// whatever the server writes next lands in the middle of our reader
    /// and every later frame is misparsed. Refuse up front rather than
    /// hand back garbage for the rest of the session.
    desynced: bool,
    /// What `initialize` said this server can answer. Without these, a
    /// server that cannot do `workspace/symbol` is indistinguishable from
    /// one that answered "nothing found", and the registry would report an
    /// honest-looking empty result for a query no server ever ran.
    supports_definition: bool,
    supports_references: bool,
    supports_workspace_symbol: bool,
}

/// What we last pushed to a server for one document.
struct OpenDoc {
    version: i32,
    /// `None` when the file's metadata could not be read, which counts as
    /// "resync every time": we cannot prove it hasn't moved.
    mtime: Option<std::time::SystemTime>,
}

/// Build the child `Command` for one LSP server. Split out of
/// [`LspClient::spawn`] for the same reason `mcp_command` is (see
/// cli/src/mcp.rs): the environment handed to a third-party binary is
/// then assertable without spawning it.
///
/// rust-analyzer et al. are third-party binaries spawned at boot
/// (cli/src/main.rs:568) with no gate in front of them; they have no
/// business holding teleia's provider keys, and unlike MCP, `LspEntry`
/// has no `env` map to grant one back with, so the scrub is
/// unconditional here.
pub(crate) fn lsp_command(entry: &LspEntry) -> Command {
    let mut cmd = Command::new(&entry.command);
    cmd.args(&entry.args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        // Null, not inherit: the TUI owns the alternate screen, and a
        // server that chats on stderr (rust-analyzer's progress) paints
        // straight into the frame. mcp.rs makes the same call.
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    teleia_tools::scrub_credentials(&mut cmd);
    cmd
}

impl LspClient {
    /// Spawn the LSP server, exchange the `initialize` handshake, send
    /// the `initialized` notification, and return a client ready for
    /// (eventually) request/response cycles.
    pub async fn spawn(name: &str, entry: &LspEntry) -> Result<Self> {
        let mut cmd = lsp_command(entry);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn LSP server `{}`", entry.command))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("LSP `{name}` exposed no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("LSP `{name}` exposed no stdout"))?;
        let mut client = Self {
            name: name.to_string(),
            server_name: None,
            server_version: None,
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            open_docs: HashMap::new(),
            desynced: false,
            supports_definition: false,
            supports_references: false,
            supports_workspace_symbol: false,
        };
        // Bound the handshake: a server that spawns but never answers
        // `initialize` (or stays stdout-silent) would otherwise hang boot
        // indefinitely, since spawn_all awaits each spawn serially. On
        // timeout the error flows into spawn_all's warnings, demoting a
        // boot hang to a `/lsps` warning.
        tokio::time::timeout(std::time::Duration::from_secs(10), client.initialize())
            .await
            .map_err(|_| anyhow!("LSP `{name}` handshake timed out after 10s"))??;
        Ok(client)
    }

    async fn initialize(&mut self) -> Result<()> {
        let root = std::env::current_dir()
            .ok()
            .and_then(|p| url_from_path(&p))
            .unwrap_or_else(|| "file:///".to_string());
        // Advertise pull-diagnostic support so capable servers
        // (rust-analyzer, pyright, tsserver, etc.) accept our
        // `textDocument/diagnostic` calls. `synchronization` is the
        // tiny capability needed to legally send `didOpen` later.
        let params = json!({
            "processId": std::process::id(),
            "rootUri": root,
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": false,
                        "didSave": false
                    },
                    "diagnostic": {
                        "dynamicRegistration": false,
                        "relatedDocumentSupport": false
                    },
                    "hover": {
                        "dynamicRegistration": false,
                        "contentFormat": ["markdown", "plaintext"]
                    },
                    // `linkSupport` deliberately on: a server that
                    // honours it answers with `LocationLink`, whose
                    // `targetSelectionRange` is the identifier's own
                    // range — so the column we print lands on the name
                    // and feeds straight back into `lsp_references`. A
                    // plain `Location` may start at `pub` or at an
                    // attribute line above it.
                    "definition": {
                        "dynamicRegistration": false,
                        "linkSupport": true
                    },
                    "references": {
                        "dynamicRegistration": false
                    }
                },
                // All 26 symbol kinds: a client that omits `valueSet` is
                // limited to 1..=18 and servers *downgrade* to fit,
                // reporting a struct as a class. `resolveSupport` is
                // deliberately absent — declaring it is the only thing
                // that lets a server reply with a uri and no range, i.e.
                // the only thing that would force a
                // `workspaceSymbol/resolve` round trip per hit.
                "workspace": {
                    "symbol": {
                        "dynamicRegistration": false,
                        "symbolKind": {
                            "valueSet": [
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26
                            ]
                        }
                    }
                }
            },
            "clientInfo": { "name": "teleia", "version": env!("CARGO_PKG_VERSION") },
        });
        let result = self.request("initialize", params).await?;
        // Pluck out serverInfo for the /lsps panel — purely
        // informational — and the three provider flags, which are not:
        // they are what makes "no server can answer this" distinguishable
        // from "every server answered, and found nothing".
        #[derive(Deserialize)]
        struct ServerInfo {
            name: String,
            #[serde(default)]
            version: Option<String>,
        }
        #[derive(Deserialize)]
        struct ServerCaps {
            #[serde(rename = "definitionProvider", default)]
            definition: Value,
            #[serde(rename = "referencesProvider", default)]
            references: Value,
            #[serde(rename = "workspaceSymbolProvider", default)]
            workspace_symbol: Value,
        }
        #[derive(Deserialize)]
        struct InitResult {
            #[serde(rename = "serverInfo", default)]
            server_info: Option<ServerInfo>,
            #[serde(default)]
            capabilities: Option<ServerCaps>,
        }
        if let Ok(init) = serde_json::from_value::<InitResult>(result) {
            if let Some(s) = init.server_info {
                self.server_name = Some(s.name);
                self.server_version = s.version;
            }
            if let Some(c) = init.capabilities {
                self.supports_definition = provider_enabled(&c.definition);
                self.supports_references = provider_enabled(&c.references);
                self.supports_workspace_symbol = provider_enabled(&c.workspace_symbol);
            }
        }
        self.notify("initialized", json!({})).await?;
        Ok(())
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        if self.desynced {
            return Err(anyhow!(
                "LSP `{}` desynchronised after a timed-out request; restart teleia to use it again",
                self.name
            ));
        }
        let id = self.next_id;
        self.next_id += 1;
        let payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.write_frame(&payload).await?;
        // Bound the wait the way `spawn` bounds the handshake. A cold
        // rust-analyzer can take tens of seconds to answer
        // `textDocument/references`; past this the whole turn is parked
        // with nothing on screen to explain it.
        let timeout = std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS);
        match tokio::time::timeout(timeout, self.await_response(id)).await {
            Ok(r) => r,
            Err(_) => {
                // The timeout cancels `await_response` mid-frame, and
                // `read_line`/`read_exact` are not cancel-safe: the bytes
                // already consumed are gone and the rest of that frame
                // would be parsed as the next one's headers. Every later
                // answer from this server would be garbage, so refuse
                // them all rather than serve one.
                self.desynced = true;
                Err(anyhow!(
                    "LSP `{}` did not answer `{method}` within {REQUEST_TIMEOUT_SECS}s",
                    self.name
                ))
            }
        }
    }

    /// Read frames until the response to `id` arrives. Split out of
    /// [`LspClient::request`] so the timeout covers only the read — a
    /// cancelled *write* would leave a half-written frame in the
    /// server's parser, desynchronising the other direction too.
    async fn await_response(&mut self, id: u64) -> Result<Value> {
        // Skip over server-initiated notifications (no `id`) until the
        // matching response arrives.
        loop {
            let msg = self.read_frame().await?;
            if is_response_to(&msg, id) {
                if let Some(err) = msg.get("error") {
                    return Err(anyhow!("LSP `{}` returned error: {err}", self.name));
                }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
            // Not our response. A server->client *request* (has `method` and
            // `id`) must be answered or a server that blocks on our reply
            // deadlocks the turn — including one whose `id` collides with
            // ours, which `is_response_to` deliberately excludes. Answer
            // MethodNotFound; plain notifications (no `id`) are ignored.
            if let Some(reply) = method_not_found_reply(&msg) {
                let _ = self.write_frame(&reply).await;
            }
        }
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let payload = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.write_frame(&payload).await
    }

    /// Push the current `text` for `uri` to the server. The first call
    /// sends `textDocument/didOpen` (version 1); every later call bumps the
    /// version and sends a full-sync `textDocument/didChange`. Without the
    /// didChange, an agentic edit-then-re-query loop would keep getting
    /// diagnostics/hover for the buffer the server opened with — stale the
    /// moment the file is edited on disk.
    pub async fn open_document(&mut self, uri: &str, language_id: &str, text: &str) -> Result<()> {
        let mtime = mtime_of(uri);
        if let Some(prev) = self.open_docs.get(uri) {
            let version = prev.version + 1;
            self.open_docs
                .insert(uri.to_string(), OpenDoc { version, mtime });
            let params = json!({
                "textDocument": { "uri": uri, "version": version },
                "contentChanges": [ { "text": text } ],
            });
            return self.notify("textDocument/didChange", params).await;
        }
        let params = json!({
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": text,
            }
        });
        self.notify("textDocument/didOpen", params).await?;
        self.open_docs
            .insert(uri.to_string(), OpenDoc { version: 1, mtime });
        Ok(())
    }

    /// Re-push `didChange` for every already-open document whose file has
    /// moved on disk since we last synced it.
    ///
    /// Diagnostics and hover never needed this: they only ever report
    /// about the one file being queried, which `open_document` has just
    /// re-synced. Definition and references are the first requests whose
    /// answers carry positions in *other* files, computed against
    /// whatever buffer this server was last handed — so without it an
    /// edit-then-re-query loop prints line numbers from before the edit.
    /// Best-effort throughout: a file that cannot be read is left as the
    /// server has it, which is no worse than not trying.
    async fn resync_open_documents(&mut self) {
        let stale: Vec<String> = self
            .open_docs
            .iter()
            .filter(|(uri, doc)| match (doc.mtime, mtime_of(uri)) {
                (Some(was), Some(now)) => was != now,
                // Either side unknown: we cannot prove it hasn't changed.
                _ => true,
            })
            .map(|(uri, _)| uri.clone())
            .collect();
        for uri in stale {
            let path = path_from_uri(&uri);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let language_id = guess_language_id(Path::new(&path));
            let _ = self.open_document(&uri, &language_id, &text).await;
        }
    }

    /// LSP 3.17 pull diagnostics. Returns rendered `path:line:col
    /// severity[source]: message` rows, one per diagnostic, or an
    /// empty string when the document is clean. Servers that don't
    /// implement `textDocument/diagnostic` surface a method-not-found
    /// error — the registry treats that as "no diagnostics from this
    /// server" and moves on.
    pub async fn pull_diagnostics(&mut self, uri: &str) -> Result<Vec<String>> {
        let params = json!({
            "textDocument": { "uri": uri }
        });
        let result = self.request("textDocument/diagnostic", params).await?;
        // Full report: { kind: "full", items: [Diagnostic...] }.
        // Unchanged report: { kind: "unchanged", resultId: "..." } — no items.
        let items = result
            .get("items")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut lines = Vec::with_capacity(items.len());
        for item in items {
            let Ok(d) = serde_json::from_value::<Diagnostic>(item) else {
                continue;
            };
            lines.push(format_diagnostic(uri, &d));
        }
        Ok(lines)
    }

    /// LSP `textDocument/hover`. `line` and `character` are 0-based
    /// (LSP wire convention). Returns the rendered hover text, or
    /// `None` when the server has nothing to say at that position
    /// (response is `null` or has empty contents). Servers that don't
    /// implement hover surface a method-not-found error — the registry
    /// treats that as "no hover from this server" and moves on.
    pub async fn hover(&mut self, uri: &str, line: u32, character: u32) -> Result<Option<String>> {
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        let result = self.request("textDocument/hover", params).await?;
        if result.is_null() {
            return Ok(None);
        }
        let Some(contents) = result.get("contents") else {
            return Ok(None);
        };
        let rendered = render_hover_contents(contents);
        Ok(if rendered.is_empty() {
            None
        } else {
            Some(rendered)
        })
    }

    /// LSP `textDocument/definition`. `line` and `character` are 0-based
    /// (LSP wire convention). Returns every position the server resolved,
    /// normalised out of the four legal result shapes. A server that
    /// didn't advertise the capability is never asked.
    async fn definition(&mut self, uri: &str, line: u32, character: u32) -> Result<Vec<Loc>> {
        if !self.supports_definition {
            return Ok(Vec::new());
        }
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character }
        });
        let result = self.request("textDocument/definition", params).await?;
        Ok(locations_from_value(&result))
    }

    /// LSP `textDocument/references`, declaration included. `line` and
    /// `character` are 0-based. The `context` object is mandatory in the
    /// spec and is not defaulted consistently — some servers reject the
    /// request without it, others silently read it as
    /// `includeDeclaration: false` — so it is always sent. The
    /// declaration is worth having: it anchors the result set and proves
    /// the position resolved to something.
    async fn references(&mut self, uri: &str, line: u32, character: u32) -> Result<Vec<Loc>> {
        if !self.supports_references {
            return Ok(Vec::new());
        }
        let params = json!({
            "textDocument": { "uri": uri },
            "position": { "line": line, "character": character },
            "context": { "includeDeclaration": true }
        });
        let result = self.request("textDocument/references", params).await?;
        Ok(locations_from_value(&result))
    }

    /// LSP `workspace/symbol`. Queries the server's index, so unlike
    /// every other request here it touches no document and needs no
    /// `didOpen`. `searchKind` is rust-analyzer's own extension: its
    /// default is types-only, which hides every free function, and
    /// servers that don't know the field ignore it.
    async fn workspace_symbols(&mut self, query: &str) -> Result<Vec<SymbolHit>> {
        if !self.supports_workspace_symbol {
            return Ok(Vec::new());
        }
        let params = json!({ "query": query, "searchKind": "allSymbols" });
        let result = self.request("workspace/symbol", params).await?;
        Ok(symbols_from_value(&result))
    }

    /// LSP framing: `Content-Length: N\r\n\r\n<N bytes of JSON>`.
    async fn write_frame(&mut self, payload: &Value) -> Result<()> {
        let body = serde_json::to_vec(payload)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        self.stdin.write_all(header.as_bytes()).await?;
        self.stdin.write_all(&body).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_frame(&mut self) -> Result<Value> {
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).await?;
            if n == 0 {
                return Err(anyhow!("LSP `{}` closed stdout", self.name));
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some((k, v)) = trimmed.split_once(':') {
                if k.eq_ignore_ascii_case("content-length") {
                    content_length = Some(v.trim().parse().with_context(|| {
                        format!("invalid Content-Length from LSP `{}`", self.name)
                    })?);
                }
            }
        }
        let n = checked_frame_len(content_length, &self.name)?;
        let mut buf = vec![0u8; n];
        self.stdout.read_exact(&mut buf).await?;
        let v: Value = serde_json::from_slice(&buf)
            .with_context(|| format!("LSP `{}` returned non-JSON body", self.name))?;
        Ok(v)
    }
}

/// Bound every post-handshake request the way `spawn` bounds the
/// handshake itself. Generous, because a cold rust-analyzer really does
/// take tens of seconds to answer the first `textDocument/references`
/// in a large workspace — but finite, because the turn is parked for
/// the whole wait.
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Row caps, per tool. `teleia-agent` trims any tool result past 12k
/// characters, keeping the head and tail and eliding the middle — which
/// would punch a hole through a location list and leave the model
/// counting rows it can no longer see. Cap here instead, in the registry,
/// where the true total is still known and can be reported.
const MAX_DEFINITION_ROWS: usize = 20;
const MAX_REFERENCE_ROWS: usize = 100;
const MAX_SYMBOL_ROWS: usize = 100;

/// The other half of the cap: stop emitting rows once the rendered block
/// passes this, so a hundred deeply-indented long lines can't reach the
/// agent's trim threshold either.
const ROW_CHAR_BUDGET: usize = 8_000;

/// Source lines are echoed into location rows; longer ones are cut with a
/// trailing `…`.
const MAX_SOURCE_LINE: usize = 120;

/// Bound the server-declared LSP frame size before allocating it, so a
/// bogus or corrupt `Content-Length` can't trigger a huge one-shot
/// allocation that OOM-aborts the process. 64 MiB is far above any real
/// LSP payload.
const MAX_LSP_FRAME: usize = 64 * 1024 * 1024;

fn checked_frame_len(content_length: Option<usize>, name: &str) -> Result<usize> {
    let n = content_length.ok_or_else(|| anyhow!("LSP `{name}` sent no Content-Length"))?;
    if n > MAX_LSP_FRAME {
        return Err(anyhow!(
            "LSP `{name}` frame too large: {n} bytes (cap {MAX_LSP_FRAME})"
        ));
    }
    Ok(n)
}

/// If `msg` is a server->client *request* (has both `method` and `id`),
/// build the JSON-RPC MethodNotFound reply that unblocks a server which
/// waits on our response. Returns `None` for responses and notifications.
fn method_not_found_reply(msg: &Value) -> Option<Value> {
    msg.get("method")?;
    let id = msg.get("id").cloned()?;
    Some(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": -32601, "message": "method not found" }
    }))
}

/// True iff `msg` is the response to our request `id`: it carries a
/// matching `id` and no `method`. A server-initiated *request* also has an
/// `id` (plus a `method`); without the `method` check, one whose id
/// happened to collide with ours would be mis-read as our response —
/// dropping the real reply and leaving the server's request unanswered.
fn is_response_to(msg: &Value, id: u64) -> bool {
    if msg.get("method").is_some() {
        return false;
    }
    // Match string-form ids too (mirrors mcp.rs): we send numeric ids,
    // but a server that echoes ours back as a JSON string would
    // otherwise never be recognized and the request loop would hang.
    match msg.get("id") {
        Some(v) => v.as_u64() == Some(id) || v.as_str() == Some(id.to_string().as_str()),
        None => false,
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Bytes that may sit in a `file:` URI path unescaped: RFC 3986's
/// unreserved set plus the separator. Everything else — spaces, `#`, `%`
/// and every non-ASCII byte — is percent-encoded.
fn uri_safe(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'/')
}

/// Convert an absolute path to a `file://` URI, percent-encoding anything
/// that would otherwise change the URI's meaning. Returns None on
/// non-absolute paths or non-utf8 segments.
///
/// The encoding was invisible while `pull_diagnostics` only ever rendered
/// the URI *we* sent. Definition, references and symbols render URIs the
/// *server* produced: one that parses ours through a real URL type echoes
/// back the normalised form, so `/home/u/my project` comes home as
/// `my%20project` — and the dedup key would split one file into two.
fn url_from_path(p: &Path) -> Option<String> {
    let p = p.to_str()?;
    if !p.starts_with('/') {
        return None;
    }
    let mut out = String::from("file://");
    for &b in p.as_bytes() {
        if uri_safe(b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    Some(out)
}

/// One percent-escape's worth of hex, or None if it isn't hex at all.
fn hex_pair(hi: u8, lo: u8) -> Option<u8> {
    let nib = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    Some(nib(hi)? * 16 + nib(lo)?)
}

/// Recover a display path from a URI, percent-decoding a `file:` one.
///
/// Anything else is returned verbatim, because it is not a path at all:
/// jdtls answers with `jar:file://…!/java/lang/String.java`, pyright with
/// `zipfile://…`, rust-analyzer with `rust-analyzer://…` for expanded
/// code. Printing those unchanged is honest; stripping a `file://` prefix
/// that isn't there and calling the remainder a path is not.
fn path_from_uri(uri: &str) -> String {
    let Some(rest) = uri.strip_prefix("file://") else {
        return uri.to_string();
    };
    let bytes = rest.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        // A `%` that isn't followed by two hex digits is a literal `%` in
        // a filename, not a broken escape — keep it as it came.
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Some(v) = hex_pair(bytes[i + 1], bytes[i + 2]) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| rest.to_string())
}

/// Last-modified time of the file a `file:` URI names, when it has one.
fn mtime_of(uri: &str) -> Option<std::time::SystemTime> {
    std::fs::metadata(path_from_uri(uri))
        .and_then(|m| m.modified())
        .ok()
}

/// True iff a `ServerCapabilities` provider field means "yes". The spec
/// types these as `boolean | XOptions`, so an options *object* — even an
/// empty one — is an affirmative, and only `false` or an absent field are
/// refusals.
fn provider_enabled(v: &Value) -> bool {
    !(v.is_null() || v == &Value::Bool(false))
}

/// Render one LSP diagnostic as `path:line:col severity[source]: msg`.
/// LSP line/column are 0-based; we add 1 to match how editors display
/// positions. `path` is derived from the URI by stripping `file://`.
fn format_diagnostic(uri: &str, d: &Diagnostic) -> String {
    let path = path_from_uri(uri);
    let line = d.range.start.line + 1;
    let col = d.range.start.character + 1;
    let sev = match d.severity {
        Some(1) => "error",
        Some(2) => "warning",
        Some(3) => "info",
        Some(4) => "hint",
        _ => "diag",
    };
    match d.source.as_deref() {
        Some(s) if !s.is_empty() => format!("{path}:{line}:{col} {sev} [{s}]: {}", d.message),
        _ => format!("{path}:{line}:{col} {sev}: {}", d.message),
    }
}

/// Flatten an LSP Hover `contents` field into a single string. The
/// spec allows three shapes: a bare string, a `{ language, value }`
/// MarkedString, a `{ kind, value }` MarkupContent, or an array of
/// MarkedStrings. We collapse the array case with blank-line
/// separators and pull `value` out of either object shape.
fn render_hover_contents(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .map(render_hover_contents)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        Value::Object(obj) => obj
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// One resolved position, 0-based exactly as LSP sends it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Loc {
    uri: String,
    line: u32,
    character: u32,
}

/// Flatten a `textDocument/definition` or `textDocument/references`
/// result into positions. The definition result alone has four legal
/// shapes — `null`, a bare `Location`, `Location[]`, `LocationLink[]` —
/// and servers send `LocationLink` whether or not the client asked for
/// it, so this matches on `Value` the way `render_hover_contents` does
/// rather than deserialising into a struct: one `serde` miss would
/// silently render the whole result as "nothing found".
fn locations_from_value(v: &Value) -> Vec<Loc> {
    match v {
        Value::Array(arr) => arr.iter().filter_map(one_location).collect(),
        Value::Object(_) => one_location(v).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// One element of such a result. Prefers `targetSelectionRange` (the
/// identifier alone) over `targetRange` (the whole item, doc comment and
/// attributes included) over `range`, so the column printed lands on the
/// name and feeds back into another `lsp_*` call.
fn one_location(v: &Value) -> Option<Loc> {
    let uri = v
        .get("uri")
        .or_else(|| v.get("targetUri"))
        .and_then(Value::as_str)?;
    let start = v
        .get("targetSelectionRange")
        .or_else(|| v.get("targetRange"))
        .or_else(|| v.get("range"))?
        .get("start")?;
    Some(Loc {
        uri: uri.to_string(),
        line: start.get("line").and_then(Value::as_u64).unwrap_or(0) as u32,
        character: start.get("character").and_then(Value::as_u64).unwrap_or(0) as u32,
    })
}

/// One `workspace/symbol` hit.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SymbolHit {
    name: String,
    kind: u8,
    container: Option<String>,
    uri: String,
    /// `None` for a 3.17 `WorkspaceSymbol` whose `location` carries only
    /// a `uri`. That form is legal only when the client declared
    /// `resolveSupport`, which we deliberately do not — but surviving one
    /// costs a branch, and dropping the hit would lose a real symbol.
    pos: Option<(u32, u32)>,
}

/// Flatten a `workspace/symbol` result. Legacy `SymbolInformation` and
/// 3.17 `WorkspaceSymbol` carry no tag distinguishing them when the
/// location is full, so both parse through one permissive path keyed on
/// whether `location.range` is there.
fn symbols_from_value(v: &Value) -> Vec<SymbolHit> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter().filter_map(one_symbol).collect()
}

fn one_symbol(v: &Value) -> Option<SymbolHit> {
    let name = v.get("name").and_then(Value::as_str)?;
    let location = v.get("location")?;
    let uri = location.get("uri").and_then(Value::as_str)?;
    let pos = location.get("range").and_then(|r| r.get("start")).map(|s| {
        (
            s.get("line").and_then(Value::as_u64).unwrap_or(0) as u32,
            s.get("character").and_then(Value::as_u64).unwrap_or(0) as u32,
        )
    });
    Some(SymbolHit {
        name: name.to_string(),
        kind: v.get("kind").and_then(Value::as_u64).unwrap_or(0) as u8,
        // An empty container is treated as absent, the same guard
        // `format_diagnostic` applies to `source`.
        container: v
            .get("containerName")
            .and_then(Value::as_str)
            .filter(|c| !c.is_empty())
            .map(str::to_string),
        uri: uri.to_string(),
        pos,
    })
}

/// LSP `SymbolKind` 1..=26 as a lowercase label, hyphenated so the column
/// stays one whitespace-free token like `error`/`warning` do. Anything
/// outside the range falls back to `symbol`, mirroring
/// `format_diagnostic`'s `_ => "diag"`: the spec requires a client that
/// declares a `valueSet` to handle values outside it gracefully.
fn symbol_kind_name(k: u8) -> &'static str {
    match k {
        1 => "file",
        2 => "module",
        3 => "namespace",
        4 => "package",
        5 => "class",
        6 => "method",
        7 => "property",
        8 => "field",
        9 => "constructor",
        10 => "enum",
        11 => "interface",
        12 => "function",
        13 => "variable",
        14 => "constant",
        15 => "string",
        16 => "number",
        17 => "boolean",
        18 => "array",
        19 => "object",
        20 => "key",
        21 => "null",
        22 => "enum-member",
        23 => "struct",
        24 => "event",
        25 => "operator",
        26 => "type-parameter",
        _ => "symbol",
    }
}

/// What [`dedupe_locations`] sorts on: the *decoded* path, so two
/// spellings of one URI collapse, then the position.
type LocKey = (String, u32, u32);

/// One hit paired with its sort key, so the key is computed once per
/// location rather than on every comparison.
type KeyedHit = (LocKey, (Loc, String));

/// Sort by (path, line, column) and drop exact repeats — the same
/// location reported by two servers that both cover the file.
///
/// The key excludes `range.end` deliberately: servers disagree about
/// extent (identifier vs whole item, `targetRange` vs
/// `targetSelectionRange`), so an end-sensitive key under-dedups and
/// doubles the count. It stops at the column just as deliberately —
/// `foo(foo())` is two real hits on one line. The sort is stable and the
/// input arrives in client order, so the first server to report a
/// location is the one credited for it.
fn dedupe_locations(hits: Vec<(Loc, String)>) -> Vec<(Loc, String)> {
    let mut keyed: Vec<KeyedHit> = hits
        .into_iter()
        .map(|(l, server)| {
            let k = (path_from_uri(&l.uri), l.line, l.character);
            (k, (l, server))
        })
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    keyed.dedup_by(|a, b| a.0 == b.0);
    keyed.into_iter().map(|(_, v)| v).collect()
}

/// Render one located result as `path:line:col label [server]: text` —
/// byte for byte the grammar `format_diagnostic` emits, space before the
/// bracket and colon after it included, so every `lsp_*` row is one
/// shape. `line`/`character` arrive 0-based from the wire and display
/// +1. A missing source line drops the suffix, not the row.
fn format_location_row(
    uri: &str,
    line: u32,
    character: u32,
    label: &str,
    server: &str,
    text: Option<&str>,
) -> String {
    let path = path_from_uri(uri);
    let (line, col) = (line + 1, character + 1);
    match text.filter(|t| !t.is_empty()) {
        Some(t) => format!("{path}:{line}:{col} {label} [{server}]: {t}"),
        None => format!("{path}:{line}:{col} {label} [{server}]"),
    }
}

/// Render one workspace symbol as
/// `path:line:col kind [server]: name (in container)`.
///
/// The container is parenthesised rather than joined with a separator
/// because one `containerName` field serves every language, and any
/// single separator — `::`, `.` — is wrong in most of them, producing a
/// string that looks pasteable and isn't.
fn format_symbol_row(hit: &SymbolHit, server: &str) -> String {
    let path = path_from_uri(&hit.uri);
    let kind = symbol_kind_name(hit.kind);
    let at = match hit.pos {
        Some((l, c)) => format!("{path}:{}:{}", l + 1, c + 1),
        None => path,
    };
    match &hit.container {
        Some(c) => format!("{at} {kind} [{server}]: {} (in {c})", hit.name),
        None => format!("{at} {kind} [{server}]: {}", hit.name),
    }
}

/// Strip indentation, flatten tabs, and cut at [`MAX_SOURCE_LINE`] with a
/// trailing `…`. The column already encodes position, so leading
/// whitespace carries nothing and costs tokens on every nested hit.
fn trim_source_line(s: &str) -> String {
    let flat = s.replace('\t', " ");
    let t = flat.trim();
    if t.chars().count() <= MAX_SOURCE_LINE {
        return t.to_string();
    }
    let cut: String = t.chars().take(MAX_SOURCE_LINE).collect();
    format!("{cut}…")
}

/// Expand over word characters around `character` to recover the
/// identifier the caller pointed at. It feeds the references header,
/// which is the only check on the dominant failure mode of a
/// position-driven tool: a column one character off the name resolves a
/// different token and answers confidently about that instead.
fn identifier_at(line_text: &str, character: u32) -> Option<String> {
    let chars: Vec<char> = line_text.chars().collect();
    let i = character as usize;
    let word = |c: char| c.is_alphanumeric() || c == '_';
    if !chars.get(i).copied().is_some_and(word) {
        return None;
    }
    let mut start = i;
    while start > 0 && word(chars[start - 1]) {
        start -= 1;
    }
    let mut end = i;
    while end + 1 < chars.len() && word(chars[end + 1]) {
        end += 1;
    }
    Some(chars[start..=end].iter().collect())
}

/// How many of `rows` to keep: the first `max_rows`, and no more than
/// `char_budget` characters' worth. Two limits because a row count alone
/// can still overrun the agent's trim, and a char budget alone leaves the
/// model with no countable boundary. Always keeps at least one row — a
/// single over-budget row is still worth more than an empty answer.
fn apply_cap(rows: &[String], max_rows: usize, char_budget: usize) -> usize {
    let mut used = 0;
    for (i, r) in rows.iter().enumerate() {
        if i >= max_rows {
            return i.max(1);
        }
        // +1 for the newline joining it to the row above.
        used += r.chars().count() + 1;
        if used > char_budget {
            return i.max(1);
        }
    }
    rows.len()
}

/// `path n, path n, …` for the busiest `top` files, then `+N more files`.
/// The cap truncates in path order, which biases towards whatever sorts
/// first; this is what tells the model where the mass actually is,
/// including in files that contributed no visible row at all.
fn file_tally(paths: &[String], top: usize) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for p in paths {
        *counts.entry(p.as_str()).or_default() += 1;
    }
    let mut v: Vec<(&str, usize)> = counts.into_iter().collect();
    // Busiest first, then by path: `HashMap` order is not stable, and a
    // tally that reshuffles between identical queries reads as churn.
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let shown: Vec<String> = v
        .iter()
        .take(top)
        .map(|(p, n)| format!("{p} {n}"))
        .collect();
    match v.len().saturating_sub(top) {
        0 => shown.join(", "),
        rest => format!("{}, +{rest} more files", shown.join(", ")),
    }
}

/// Guess an LSP language id from a file extension. Falls through to
/// the extension itself for anything we don't have a canonical id for —
/// modern language servers tend to accept the extension as the id.
fn guess_language_id(p: &Path) -> String {
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "rs" => "rust".into(),
        "ts" => "typescript".into(),
        "tsx" => "typescriptreact".into(),
        "js" => "javascript".into(),
        "jsx" => "javascriptreact".into(),
        "py" => "python".into(),
        "go" => "go".into(),
        "lua" => "lua".into(),
        "c" | "h" => "c".into(),
        "cpp" | "cxx" | "cc" | "hpp" => "cpp".into(),
        "java" => "java".into(),
        "rb" => "ruby".into(),
        "sh" | "bash" => "shellscript".into(),
        "" => "plaintext".into(),
        other => other.into(),
    }
}

/// Resolve a caller-supplied path into the (absolute path, URI, text,
/// language id) prologue every document-scoped request shares.
fn document_context(path: &str) -> Result<(std::path::PathBuf, String, String, String)> {
    let abs = Path::new(path)
        .canonicalize()
        .with_context(|| format!("resolving `{path}`"))?;
    let uri =
        url_from_path(&abs).ok_or_else(|| anyhow!("could not form file:// URI for `{path}`"))?;
    let text = std::fs::read_to_string(&abs).with_context(|| format!("reading `{path}`"))?;
    let language_id = guess_language_id(&abs);
    Ok((abs, uri, text, language_id))
}

/// Render located results, echoing the source line each one points at so
/// the model rarely has to `read` the file afterwards. Files are read
/// once and cached: a references result routinely names one file dozens
/// of times, and a file that cannot be read costs its rows the suffix,
/// not their existence.
fn render_location_rows(hits: &[(Loc, String)], label: &str) -> Vec<String> {
    let mut cache: HashMap<String, Option<Vec<String>>> = HashMap::new();
    hits.iter()
        .map(|(loc, server)| {
            let path = path_from_uri(&loc.uri);
            let lines = cache.entry(path.clone()).or_insert_with(|| {
                std::fs::read_to_string(&path)
                    .ok()
                    .map(|t| t.lines().map(str::to_string).collect())
            });
            let text = lines
                .as_ref()
                .and_then(|ls| ls.get(loc.line as usize))
                .map(|l| trim_source_line(l));
            format_location_row(
                &loc.uri,
                loc.line,
                loc.character,
                label,
                server,
                text.as_deref(),
            )
        })
        .collect()
}

/// The argument schema shared by every position-taking `lsp_*` tool.
/// One function rather than three literals so a wording fix to the
/// position contract cannot reach two of them and miss the third.
fn position_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file (absolute or relative to teleia's cwd)"
            },
            "line": {
                "type": "integer",
                "description": "1-based line number (matches any lsp_* row)"
            },
            "character": {
                "type": "integer",
                "description": "1-based column number (matches any lsp_* row)"
            }
        },
        "required": ["path", "line", "character"]
    })
}

/// Whether `dir` — or any directory above it — looks like a project
/// this server should be started for.
///
/// An empty pattern list matches everything, which is what a config that
/// never set one has always meant. Patterns are shell globs tested
/// against file names, so both `"Cargo.toml"` and `"*.csproj"` work; one
/// that fails to parse as a glob still matches as a literal name. The
/// walk goes upward because teleia is routinely launched from a
/// subdirectory of the workspace its servers serve.
pub fn root_matches(dir: &Path, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    let globs: Vec<glob::Pattern> = patterns
        .iter()
        .filter(|p| p.contains(['*', '?', '[']))
        .filter_map(|p| glob::Pattern::new(p).ok())
        .collect();
    for ancestor in dir.ancestors() {
        // Literal names are the common case and answer without listing
        // the directory at all.
        if patterns.iter().any(|p| ancestor.join(p).exists()) {
            return true;
        }
        if globs.is_empty() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(ancestor) else {
            continue;
        };
        for e in entries.flatten() {
            let name = e.file_name();
            if globs.iter().any(|g| g.matches(&name.to_string_lossy())) {
                return true;
            }
        }
    }
    false
}

/// Set of running LSP clients. Used by the TUI's `/lsps` panel — for
/// now the registry just owns the clients so they stay alive (the LSP
/// children are killed on Drop via `kill_on_drop`) and exposes a
/// formatted summary.
pub struct LspRegistry {
    clients: Vec<LspClient>,
    /// Boot-time warnings (spawn failures). Surfaced via the `/lsps`
    /// panel instead of stderr so loading stays silent at the
    /// terminal level.
    warnings: Vec<String>,
}

impl LspRegistry {
    /// Spawn every configured LSP server. Failures don't abort
    /// startup — teleia keeps booting and the error is stashed in
    /// `warnings()` for the `/lsps` panel.
    ///
    /// `on_step` is invoked once per entry with `(name, index, total)`
    /// (1-based index) before that entry's spawn begins, so the boot
    /// splash can report which server is currently being contacted.
    pub async fn spawn_all<'a, I, F>(entries: I, mut on_step: F) -> Self
    where
        I: IntoIterator<Item = (&'a String, &'a LspEntry)>,
        F: FnMut(&str, usize, usize),
    {
        let entries: Vec<(&'a String, &'a LspEntry)> = entries.into_iter().collect();
        let total = entries.len();
        let mut clients = Vec::new();
        let mut warnings = Vec::new();
        for (i, (name, entry)) in entries.into_iter().enumerate() {
            on_step(name, i + 1, total);
            match LspClient::spawn(name, entry).await {
                Ok(c) => clients.push(c),
                Err(e) => warnings.push(format!("LSP `{name}` failed to start: {e:#}")),
            }
        }
        Self { clients, warnings }
    }

    /// Boot-time warnings collected by `spawn_all`. Empty on a clean boot.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Per-server (name, serverInfo, version) — drives the `/lsps`
    /// panel rendering.
    pub fn server_summaries(&self) -> Vec<(String, Option<String>, Option<String>)> {
        self.clients
            .iter()
            .map(|c| {
                (
                    c.name.clone(),
                    c.server_name.clone(),
                    c.server_version.clone(),
                )
            })
            .collect()
    }

    /// Fan a pull-diagnostic request across every running server.
    /// Each client gets a `didOpen` on first sight, then the pull.
    /// Per-server failures (method-not-found, connection drops) are
    /// swallowed — they just mean "no diagnostics from that server".
    pub async fn diagnostics_for(&mut self, path: &str) -> Result<String> {
        let (abs, uri, text, language_id) = document_context(path)?;
        let mut all: Vec<String> = Vec::new();
        for client in self.clients.iter_mut() {
            // Best-effort per server: ignore didOpen failures (some
            // servers refuse languages they don't recognise).
            if client
                .open_document(&uri, &language_id, &text)
                .await
                .is_err()
            {
                continue;
            }
            if let Ok(lines) = client.pull_diagnostics(&uri).await {
                all.extend(lines);
            }
        }
        if all.is_empty() {
            Ok(format!("no diagnostics for {}", abs.display()))
        } else {
            Ok(all.join("\n"))
        }
    }

    /// Fan a hover request across every running server. Positions are
    /// 1-based on the way in (to match how diagnostics render
    /// `path:line:col`) and converted to LSP's 0-based wire form
    /// inside. Per-server failures are swallowed the same way as
    /// `diagnostics_for`. Hover blobs from multiple servers are joined
    /// with a `---` separator.
    pub async fn hover_for(&mut self, path: &str, line: u32, character: u32) -> Result<String> {
        let (abs, uri, text, language_id) = document_context(path)?;
        let lsp_line = line.saturating_sub(1);
        let lsp_char = character.saturating_sub(1);

        let mut blobs: Vec<String> = Vec::new();
        for client in self.clients.iter_mut() {
            if client
                .open_document(&uri, &language_id, &text)
                .await
                .is_err()
            {
                continue;
            }
            if let Ok(Some(h)) = client.hover(&uri, lsp_line, lsp_char).await {
                blobs.push(h);
            }
        }
        if blobs.is_empty() {
            Ok(format!(
                "no hover info at {}:{}:{}",
                abs.display(),
                line,
                character
            ))
        } else {
            Ok(blobs.join("\n\n---\n\n"))
        }
    }

    /// Fan a definition request across every running server. Same
    /// prologue as `diagnostics_for`, then a `didOpen`/`didChange` per
    /// client before the request. Locations from different servers are
    /// deduped rather than concatenated the way hover blobs are: two
    /// hover blobs are two views of one symbol, two identical location
    /// rows are just noise.
    ///
    /// Never `Err` on an empty result — `teleia-agent` renders an `Err`
    /// as `error: …`, and a false error can trip its repeated-failure
    /// stop on a question that was answered perfectly well with "no".
    pub async fn definition_for(
        &mut self,
        path: &str,
        line: u32,
        character: u32,
    ) -> Result<String> {
        if !self.clients.iter().any(|c| c.supports_definition) {
            return Ok("no language server advertises textDocument/definition".to_string());
        }
        let (abs, uri, text, language_id) = document_context(path)?;
        let (l, c) = (line.saturating_sub(1), character.saturating_sub(1));

        let mut hits: Vec<(Loc, String)> = Vec::new();
        for client in self.clients.iter_mut() {
            if client
                .open_document(&uri, &language_id, &text)
                .await
                .is_err()
            {
                continue;
            }
            if let Ok(locs) = client.definition(&uri, l, c).await {
                let server = client.name.clone();
                hits.extend(locs.into_iter().map(|loc| (loc, server.clone())));
            }
        }
        let hits = dedupe_locations(hits);
        if hits.is_empty() {
            return Ok(format!(
                "no definition at {}:{}:{}",
                abs.display(),
                line,
                character
            ));
        }
        let total = hits.len();
        let rows = render_location_rows(&hits, "definition");
        let keep = apply_cap(&rows, MAX_DEFINITION_ROWS, ROW_CHAR_BUDGET);
        let mut out: Vec<String> = rows[..keep].to_vec();
        if keep < total {
            out.push(format!("[truncated at {keep} of {total} definitions]"));
        }
        Ok(out.join("\n"))
    }

    /// Fan a references request across every running server, resyncing
    /// stale buffers first: this answer names files other than the one
    /// being queried, so any of them the agent opened earlier and has
    /// since edited would report pre-edit line numbers.
    pub async fn references_for(
        &mut self,
        path: &str,
        line: u32,
        character: u32,
    ) -> Result<String> {
        if !self.clients.iter().any(|c| c.supports_references) {
            return Ok("no language server advertises textDocument/references".to_string());
        }
        let (abs, uri, text, language_id) = document_context(path)?;
        let (l, c) = (line.saturating_sub(1), character.saturating_sub(1));

        let mut hits: Vec<(Loc, String)> = Vec::new();
        for client in self.clients.iter_mut() {
            client.resync_open_documents().await;
            if client
                .open_document(&uri, &language_id, &text)
                .await
                .is_err()
            {
                continue;
            }
            if let Ok(locs) = client.references(&uri, l, c).await {
                let server = client.name.clone();
                hits.extend(locs.into_iter().map(|loc| (loc, server.clone())));
            }
        }
        let hits = dedupe_locations(hits);
        if hits.is_empty() {
            return Ok(format!(
                "no references at {}:{}:{}",
                abs.display(),
                line,
                character
            ));
        }

        let total = hits.len();
        let paths: Vec<String> = hits
            .iter()
            .map(|(loc, _)| path_from_uri(&loc.uri))
            .collect();
        let files = paths.iter().collect::<BTreeSet<_>>().len();
        // Name the symbol in the header. A column one character off the
        // identifier resolves a *different* token and answers confidently
        // about that instead; printing what we actually asked about is
        // the only check the model gets on that.
        let symbol = text
            .lines()
            .nth(l as usize)
            .and_then(|t| identifier_at(t, c));
        let at = format!("{}:{}:{}", abs.display(), line, character);
        let header = match &symbol {
            Some(name) => format!("{total} references to `{name}` from {at} in {files} files:"),
            None => format!("{total} references from {at} in {files} files:"),
        };

        let rows = render_location_rows(&hits, "reference");
        let keep = apply_cap(&rows, MAX_REFERENCE_ROWS, ROW_CHAR_BUDGET);
        let mut out = vec![header];
        out.extend(rows[..keep].iter().cloned());
        if keep < total {
            out.push(format!(
                "[truncated at {keep} of {total} references — {}; rows above are the first {keep} in path order]",
                file_tally(&paths, 3)
            ));
        }
        Ok(out.join("\n"))
    }

    /// Fan a workspace symbol search across every running server. No
    /// path, so no canonicalize, no read and no `didOpen` — the request
    /// goes to the server's index. Each server's own ranking is kept and
    /// the servers concatenated in configuration order: rust-analyzer and
    /// gopls fuzzy-rank, so the exact match comes first, and re-sorting
    /// by path would throw that away.
    pub async fn symbols_for(&mut self, query: &str) -> Result<String> {
        if !self.clients.iter().any(|c| c.supports_workspace_symbol) {
            return Ok("no language server advertises workspace/symbol".to_string());
        }
        let mut rows: Vec<String> = Vec::new();
        for client in self.clients.iter_mut() {
            let Ok(hits) = client.workspace_symbols(query).await else {
                continue;
            };
            let server = client.name.clone();
            rows.extend(hits.iter().map(|h| format_symbol_row(h, &server)));
        }
        if rows.is_empty() {
            return Ok(format!("no symbols matching \"{query}\""));
        }
        let total = rows.len();
        let keep = apply_cap(&rows, MAX_SYMBOL_ROWS, ROW_CHAR_BUDGET);
        let mut out: Vec<String> = rows[..keep].to_vec();
        if keep < total {
            out.push(format!(
                "[truncated at {keep} of {total} symbols matching \"{query}\" — narrow the query]"
            ));
        }
        Ok(out.join("\n"))
    }
}

impl ToolRouter for LspRegistry {
    fn definitions(&self) -> Vec<ToolDef> {
        if self.clients.is_empty() {
            return Vec::new();
        }
        let n = self.clients.len();
        let diagnostics_description = format!(
            "Get diagnostics (errors, warnings) for a source file from the \
             {n} configured language server(s). Sends textDocument/didOpen \
             on first use, then issues a pull-diagnostic request. Returns \
             one `<path>:<line>:<col> <severity> [<source>]: <message>` row \
             per diagnostic, or `no diagnostics for <path>` when clean.",
        );
        let hover_description = format!(
            "Get hover info (type, signature, doc-comment) at a source \
             position from the {n} configured language server(s). `line` \
             and `character` are 1-based and interchangeable with every \
             other `lsp_*` row — feed any printed `<path>:<line>:<col>` \
             straight back in. Returns the rendered \
             hover blob (markdown when the server provides it), or `no \
             hover info at <path>:<line>:<col>` when the server has \
             nothing at that position.",
        );
        let definition_description = format!(
            "Jump to where the symbol at a source position is defined, \
             via textDocument/definition on the {n} configured language \
             server(s). `line` and `character` are 1-based and \
             interchangeable with every other `lsp_*` row. Returns one \
             `<path>:<line>:<col> definition [<server>]: <source line>` \
             row per definition — usually one, several for a trait method \
             with impls — with the source line included so you rarely need \
             to read the file afterwards. Returns `no definition at \
             <path>:<line>:<col>` when no server resolves a symbol there; \
             a column one character off the identifier looks identical to \
             a symbol that does not exist, so confirm with `lsp_hover` \
             before concluding.",
        );
        let references_description = format!(
            "Find every use of the symbol at a source position — callers \
             of a function, readers of a field, impls of a trait method — \
             via textDocument/references on the {n} configured language \
             server(s), declaration included. Unlike `grep` this is \
             resolved by the compiler: same-named symbols in other scopes, \
             and mentions in comments and strings, are not returned. \
             `line` and `character` are 1-based and interchangeable with \
             every other `lsp_*` row. Returns a header line, then one \
             `<path>:<line>:<col> reference [<server>]: <source line>` row \
             per use sorted by path then position, capped at {MAX_REFERENCE_ROWS} \
             rows; past the cap a closing `[truncated …]` line gives the \
             true per-file counts. Returns `no references at \
             <path>:<line>:<col>` when nothing uses it — which is also \
             what an unresolved position returns, so confirm with \
             `lsp_hover` before deleting anything as dead code.",
        );
        let symbols_description = format!(
            "Search the whole workspace's symbol index by name, via \
             workspace/symbol on the {n} configured language server(s) — \
             the fastest way to locate a type, function or method when you \
             know roughly what it is called but not where it lives. \
             `query` is a name fragment matched by the server (fuzzy or \
             substring), not a regex and not a full-text search; use \
             `grep` for that. Returns one `<path>:<line>:<col> <kind> \
             [<server>]: <name> (in <container>)` row per symbol in the \
             server's own ranking order, best match first, where `<kind>` \
             is the LSP symbol kind lowercased — function, method, struct, \
             class, field, constant, module — capped at {MAX_SYMBOL_ROWS} rows. Every \
             printed position feeds straight into `lsp_definition`, \
             `lsp_references` or `lsp_hover`. Returns `no symbols matching \
             \"<query>\"` when nothing matches; early in a session that can \
             also mean the server has not finished indexing, in which case \
             fall back to `grep` rather than retrying.",
        );
        vec![
            ToolDef::new(
                DIAGNOSTICS_TOOL,
                diagnostics_description,
                json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to diagnose (absolute or relative to teleia's cwd)"
                        }
                    },
                    "required": ["path"]
                }),
            ),
            ToolDef::new(
                HOVER_TOOL,
                hover_description,
                json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file (absolute or relative to teleia's cwd)"
                        },
                        "line": {
                            "type": "integer",
                            "description": "1-based line number (matches any lsp_* row)"
                        },
                        "character": {
                            "type": "integer",
                            "description": "1-based column number (matches any lsp_* row)"
                        }
                    },
                    "required": ["path", "line", "character"]
                }),
            ),
            ToolDef::new(DEFINITION_TOOL, definition_description, position_schema()),
            ToolDef::new(REFERENCES_TOOL, references_description, position_schema()),
            ToolDef::new(
                SYMBOLS_TOOL,
                symbols_description,
                json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Symbol name or a fragment of one, e.g. `pull_diagnostics` or `Registry`. Matched by the server (fuzzy or substring) — not a regex, not a full-text search"
                        }
                    },
                    "required": ["query"]
                }),
            ),
        ]
    }
    fn handles(&self, name: &str) -> bool {
        LSP_TOOLS.contains(&name) && !self.clients.is_empty()
    }
    fn dispatch<'a>(&'a mut self, name: &'a str, args: &'a str) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let v: Value = serde_json::from_str(args)
                .with_context(|| format!("invalid JSON args for `{name}`"))?;
            // `path` is per-tool, not per-router: `lsp_symbols` queries
            // the server's index and takes a `query` instead. Hoisting the
            // extraction would fail every `lsp_symbols` call by blaming an
            // argument the model was right not to send — and that message
            // shape is exactly what `incomplete_tool_args` matches, so the
            // agent would attach a dropped-argument hint and keep retrying
            // an unwinnable call until its identical-failure stop fires.
            let path = || -> Result<String> {
                v.get("path")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("`{name}` requires a string `path` argument"))
            };
            let pos = || -> Result<(u32, u32)> {
                let line = v
                    .get("line")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("`{name}` requires an integer `line` argument"))?;
                let character = v
                    .get("character")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| anyhow!("`{name}` requires an integer `character` argument"))?;
                Ok((line as u32, character as u32))
            };
            match name {
                DIAGNOSTICS_TOOL => self.diagnostics_for(&path()?).await,
                HOVER_TOOL => {
                    let (l, c) = pos()?;
                    self.hover_for(&path()?, l, c).await
                }
                DEFINITION_TOOL => {
                    let (l, c) = pos()?;
                    self.definition_for(&path()?, l, c).await
                }
                REFERENCES_TOOL => {
                    let (l, c) = pos()?;
                    self.references_for(&path()?, l, c).await
                }
                SYMBOLS_TOOL => {
                    let q = v
                        .get("query")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("`{name}` requires a string `query` argument"))?;
                    // rust-analyzer answers an empty query by dumping a
                    // slice of its whole index — refuse rather than flood.
                    if q.trim().is_empty() {
                        return Err(anyhow!("`{name}` requires a non-empty `query`"));
                    }
                    self.symbols_for(q).await
                }
                _ => Err(anyhow!("LSP tool `{name}` not registered")),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(line: u32, col: u32, sev: u8, msg: &str, source: Option<&str>) -> Diagnostic {
        Diagnostic {
            range: Range {
                start: Position {
                    line,
                    character: col,
                },
                end: Position {
                    line,
                    character: col,
                },
            },
            severity: Some(sev),
            message: msg.into(),
            source: source.map(String::from),
        }
    }

    #[test]
    fn format_diagnostic_with_source_includes_brackets() {
        let d = diag(9, 4, 1, "no method foo", Some("rustc"));
        let out = format_diagnostic("file:///work/src/main.rs", &d);
        // LSP is 0-based, displayed 1-based → 10:5.
        assert_eq!(out, "/work/src/main.rs:10:5 error [rustc]: no method foo");
    }

    #[test]
    fn checked_frame_len_rejects_oversized_and_missing() {
        assert_eq!(checked_frame_len(Some(1024), "srv").unwrap(), 1024);
        assert!(checked_frame_len(Some(MAX_LSP_FRAME + 1), "srv").is_err());
        assert!(checked_frame_len(None, "srv").is_err());
    }

    #[test]
    fn method_not_found_reply_only_for_server_requests() {
        // server->client request (method + id) → MethodNotFound reply
        let req = json!({ "jsonrpc": "2.0", "id": 7, "method": "workspace/configuration" });
        let reply = method_not_found_reply(&req).expect("server request must be answered");
        assert_eq!(reply["id"], json!(7));
        assert_eq!(reply["error"]["code"], json!(-32601));
        // notification (method, no id) → ignored
        assert!(method_not_found_reply(&json!({ "method": "window/logMessage" })).is_none());
        // response (id, no method) → ignored
        assert!(method_not_found_reply(&json!({ "id": 3, "result": {} })).is_none());
    }

    #[test]
    fn is_response_to_excludes_colliding_server_request() {
        // Our response: matching id, no method.
        assert!(is_response_to(&json!({ "id": 4, "result": {} }), 4));
        // Different id → not ours.
        assert!(!is_response_to(&json!({ "id": 5, "result": {} }), 4));
        // A server->client request whose id collides with ours: must NOT
        // be mistaken for our response (it has a `method`).
        assert!(!is_response_to(
            &json!({ "id": 4, "method": "workspace/configuration" }),
            4
        ));
        // A notification (no id) is never our response.
        assert!(!is_response_to(
            &json!({ "method": "window/logMessage" }),
            4
        ));
        // A server that echoes our id as a JSON-RPC string id still matches —
        // otherwise the request loop would never recognize it and hang.
        assert!(is_response_to(&json!({ "id": "4", "result": {} }), 4));
        assert!(!is_response_to(&json!({ "id": "5", "result": {} }), 4));
    }

    #[test]
    fn format_diagnostic_without_source_omits_brackets() {
        let d = diag(0, 0, 2, "unused", None);
        let out = format_diagnostic("file:///x.rs", &d);
        assert_eq!(out, "/x.rs:1:1 warning: unused");
    }

    #[test]
    fn format_diagnostic_severity_table() {
        for (sev, expected) in [(1, "error"), (2, "warning"), (3, "info"), (4, "hint")] {
            let d = diag(0, 0, sev, "x", None);
            let out = format_diagnostic("file:///f", &d);
            assert!(
                out.contains(expected),
                "severity {sev} → missing `{expected}` in `{out}`"
            );
        }
    }

    #[test]
    fn guess_language_id_table() {
        assert_eq!(guess_language_id(Path::new("a.rs")), "rust");
        assert_eq!(guess_language_id(Path::new("a.py")), "python");
        assert_eq!(guess_language_id(Path::new("a.tsx")), "typescriptreact");
        assert_eq!(guess_language_id(Path::new("a.cpp")), "cpp");
        // Unknown extension falls through to the extension itself.
        assert_eq!(guess_language_id(Path::new("a.zig")), "zig");
        // No extension → plaintext sentinel.
        assert_eq!(guess_language_id(Path::new("Makefile")), "plaintext");
    }

    #[test]
    fn render_hover_handles_bare_string() {
        let v = json!("hello world");
        assert_eq!(render_hover_contents(&v), "hello world");
    }

    #[test]
    fn render_hover_handles_markup_content() {
        let v = json!({ "kind": "markdown", "value": "```rs\nfn foo()\n```" });
        assert_eq!(render_hover_contents(&v), "```rs\nfn foo()\n```");
    }

    #[test]
    fn render_hover_handles_marked_string_object() {
        let v = json!({ "language": "rust", "value": "fn foo()" });
        assert_eq!(render_hover_contents(&v), "fn foo()");
    }

    #[test]
    fn render_hover_joins_array_with_blank_lines() {
        let v = json!([
            "Type: `u32`",
            { "language": "rust", "value": "fn foo()" },
            { "kind": "markdown", "value": "Computes the answer." }
        ]);
        assert_eq!(
            render_hover_contents(&v),
            "Type: `u32`\n\nfn foo()\n\nComputes the answer."
        );
    }

    #[test]
    fn render_hover_skips_empty_array_entries() {
        let v = json!(["", { "value": "" }, "real text"]);
        assert_eq!(render_hover_contents(&v), "real text");
    }

    #[test]
    fn render_hover_object_without_value_is_empty() {
        let v = json!({ "language": "rust" });
        assert_eq!(render_hover_contents(&v), "");
    }

    fn loc(uri: &str, line: u32, character: u32) -> Loc {
        Loc {
            uri: uri.into(),
            line,
            character,
        }
    }

    #[test]
    fn format_location_row_matches_the_diagnostic_row_grammar() {
        // Same shape as `format_diagnostic`: space before the bracket,
        // colon after it, 0-based wire position displayed +1. Every
        // `lsp_*` row has to be one grammar or the model has to learn two.
        let out = format_location_row(
            "file:///work/src/lsp.rs",
            9,
            4,
            "definition",
            "rust",
            Some("pub async fn hover()"),
        );
        assert_eq!(
            out,
            "/work/src/lsp.rs:10:5 definition [rust]: pub async fn hover()"
        );
    }

    #[test]
    fn format_location_row_omits_the_text_suffix_when_the_source_line_is_missing() {
        // A file we cannot read costs its rows the echoed line, not the
        // rows themselves — the position is the part that matters.
        let out = format_location_row("file:///x.rs", 0, 0, "reference", "rust", None);
        assert_eq!(out, "/x.rs:1:1 reference [rust]");
        let empty = format_location_row("file:///x.rs", 0, 0, "reference", "rust", Some(""));
        assert_eq!(empty, "/x.rs:1:1 reference [rust]");
    }

    #[test]
    fn symbol_kind_name_table() {
        for (k, expected) in [
            (1, "file"),
            (6, "method"),
            (12, "function"),
            (14, "constant"),
            (22, "enum-member"),
            (23, "struct"),
            (26, "type-parameter"),
        ] {
            assert_eq!(symbol_kind_name(k), expected, "kind {k}");
        }
        // Outside 1..=26 — the spec requires a client declaring a
        // valueSet to survive a value outside it.
        assert_eq!(symbol_kind_name(0), "symbol");
        assert_eq!(symbol_kind_name(99), "symbol");
        // Every label is one whitespace-free token, like `error`/`warning`,
        // so the column stays parseable.
        for k in 0..=27u8 {
            assert!(!symbol_kind_name(k).contains(' '), "kind {k}");
        }
    }

    #[test]
    fn format_symbol_row_omits_an_empty_container() {
        let mut hit = SymbolHit {
            name: "hover".into(),
            kind: 6,
            container: Some("LspClient".into()),
            uri: "file:///work/lsp.rs".into(),
            pos: Some((294, 17)),
        };
        assert_eq!(
            format_symbol_row(&hit, "rust"),
            "/work/lsp.rs:295:18 method [rust]: hover (in LspClient)"
        );
        hit.container = None;
        assert_eq!(
            format_symbol_row(&hit, "rust"),
            "/work/lsp.rs:295:18 method [rust]: hover"
        );
    }

    #[test]
    fn format_symbol_row_without_a_range_drops_the_line_and_column() {
        // A 3.17 `WorkspaceSymbol` may carry only a uri. Better a row
        // naming the file than no row at all.
        let hit = SymbolHit {
            name: "Registry".into(),
            kind: 23,
            container: None,
            uri: "file:///work/lsp.rs".into(),
            pos: None,
        };
        assert_eq!(
            format_symbol_row(&hit, "rust"),
            "/work/lsp.rs struct [rust]: Registry"
        );
    }

    #[test]
    fn locations_from_value_handles_all_four_definition_shapes() {
        // `null`, a bare Location, Location[], LocationLink[] — all four
        // are legal answers to one request, and a server picks without
        // asking.
        assert!(locations_from_value(&Value::Null).is_empty());

        let bare =
            json!({ "uri": "file:///a.rs", "range": { "start": { "line": 3, "character": 7 } } });
        assert_eq!(locations_from_value(&bare), vec![loc("file:///a.rs", 3, 7)]);

        let list = json!([
            { "uri": "file:///a.rs", "range": { "start": { "line": 1, "character": 2 } } },
            { "uri": "file:///b.rs", "range": { "start": { "line": 4, "character": 0 } } }
        ]);
        assert_eq!(
            locations_from_value(&list),
            vec![loc("file:///a.rs", 1, 2), loc("file:///b.rs", 4, 0)]
        );

        let links = json!([{
            "targetUri": "file:///c.rs",
            "targetRange": { "start": { "line": 9, "character": 0 } },
            "targetSelectionRange": { "start": { "line": 11, "character": 7 } }
        }]);
        assert_eq!(
            locations_from_value(&links),
            vec![loc("file:///c.rs", 11, 7)]
        );
    }

    #[test]
    fn locations_from_value_prefers_the_identifier_range() {
        // `targetRange` covers the whole item, doc comment and attributes
        // included, so it points at a blank line above the name as often
        // as not. `targetSelectionRange` is the name itself, which is
        // what has to round-trip back into `lsp_references`.
        let link = json!([{
            "targetUri": "file:///c.rs",
            "targetRange": { "start": { "line": 9, "character": 0 } },
            "targetSelectionRange": { "start": { "line": 11, "character": 7 } }
        }]);
        assert_eq!(locations_from_value(&link)[0].line, 11);

        // With no selection range, fall back rather than drop the hit.
        let no_sel = json!([{
            "targetUri": "file:///c.rs",
            "targetRange": { "start": { "line": 9, "character": 0 } }
        }]);
        assert_eq!(locations_from_value(&no_sel)[0].line, 9);
    }

    #[test]
    fn symbols_from_value_reads_both_result_types() {
        // Legacy `SymbolInformation` and 3.17 `WorkspaceSymbol` are not
        // distinguishable by any tag when the location is full.
        let v = json!([
            {
                "name": "pull_diagnostics",
                "kind": 12,
                "containerName": "lsp",
                "location": {
                    "uri": "file:///a.rs",
                    "range": { "start": { "line": 5, "character": 3 } }
                }
            },
            {
                "name": "Registry",
                "kind": 23,
                "location": { "uri": "file:///b.rs" }
            }
        ]);
        let hits = symbols_from_value(&v);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].name, "pull_diagnostics");
        assert_eq!(hits[0].pos, Some((5, 3)));
        assert_eq!(hits[0].container.as_deref(), Some("lsp"));
        // Range-less: survives, with no position rather than a fake one.
        assert_eq!(hits[1].pos, None);
        assert_eq!(hits[1].container, None);
    }

    #[test]
    fn symbols_from_value_treats_an_empty_container_as_absent() {
        let v = json!([{
            "name": "main", "kind": 12, "containerName": "",
            "location": { "uri": "file:///a.rs" }
        }]);
        assert_eq!(symbols_from_value(&v)[0].container, None);
    }

    #[test]
    fn dedupe_locations_keys_on_the_start_not_the_extent() {
        // Two servers report one definition, disagreeing about how far it
        // extends. An end-sensitive key would under-dedup and double the
        // count of everything a second server also covers.
        let hits = vec![
            (loc("file:///a.rs", 3, 7), "rust".to_string()),
            (loc("file:///a.rs", 3, 7), "other".to_string()),
        ];
        let out = dedupe_locations(hits);
        assert_eq!(out.len(), 1);
        // First server to report it is the one credited.
        assert_eq!(out[0].1, "rust");
    }

    #[test]
    fn dedupe_locations_keeps_two_hits_on_one_line() {
        // `foo(foo())` is two real references. A path+line key would
        // silently collapse them.
        let hits = vec![
            (loc("file:///a.rs", 3, 0), "rust".to_string()),
            (loc("file:///a.rs", 3, 4), "rust".to_string()),
        ];
        assert_eq!(dedupe_locations(hits).len(), 2);
    }

    #[test]
    fn dedupe_locations_sorts_by_path_then_line_then_column() {
        let hits = vec![
            (loc("file:///b.rs", 1, 0), "rust".to_string()),
            (loc("file:///a.rs", 9, 2), "rust".to_string()),
            (loc("file:///a.rs", 9, 1), "rust".to_string()),
            (loc("file:///a.rs", 2, 0), "rust".to_string()),
        ];
        let out: Vec<(String, u32, u32)> = dedupe_locations(hits)
            .into_iter()
            .map(|(l, _)| (path_from_uri(&l.uri), l.line, l.character))
            .collect();
        assert_eq!(
            out,
            vec![
                ("/a.rs".to_string(), 2, 0),
                ("/a.rs".to_string(), 9, 1),
                ("/a.rs".to_string(), 9, 2),
                ("/b.rs".to_string(), 1, 0),
            ]
        );
    }

    #[test]
    fn dedupe_locations_matches_two_spellings_of_one_path() {
        // One server echoes the URI we sent, another re-encodes it
        // through a real URL type. Both name the same file, and the key
        // is the decoded path so both collapse to one row.
        let hits = vec![
            (loc("file:///work/my%20dir/a.rs", 1, 0), "rust".to_string()),
            (loc("file:///work/my dir/a.rs", 1, 0), "other".to_string()),
        ];
        assert_eq!(dedupe_locations(hits).len(), 1);
    }

    #[test]
    fn identifier_at_expands_over_word_characters() {
        assert_eq!(
            identifier_at("let foo_bar = 1;", 6).as_deref(),
            Some("foo_bar")
        );
        // From the first and last character of the name, not just inside.
        assert_eq!(
            identifier_at("let foo_bar = 1;", 4).as_deref(),
            Some("foo_bar")
        );
        assert_eq!(
            identifier_at("let foo_bar = 1;", 10).as_deref(),
            Some("foo_bar")
        );
        assert_eq!(identifier_at("a.method2()", 2).as_deref(), Some("method2"));
    }

    #[test]
    fn identifier_at_returns_none_off_an_identifier() {
        assert_eq!(identifier_at("let x = 1;", 3), None); // space
        assert_eq!(identifier_at("let x = 1;", 6), None); // `=`
        assert_eq!(identifier_at("let x = 1;", 99), None); // past the end
        assert_eq!(identifier_at("", 0), None);
    }

    #[test]
    fn trim_source_line_strips_indentation_and_caps_length() {
        assert_eq!(trim_source_line("\t    let x = 1;  "), "let x = 1;");
        let long = "x".repeat(MAX_SOURCE_LINE + 40);
        let cut = trim_source_line(&long);
        assert_eq!(cut.chars().count(), MAX_SOURCE_LINE + 1);
        assert!(cut.ends_with('…'));
        // Exactly at the cap is not truncated.
        let exact = "y".repeat(MAX_SOURCE_LINE);
        assert_eq!(trim_source_line(&exact), exact);
    }

    #[test]
    fn url_from_path_percent_encodes_what_would_change_the_uri() {
        assert_eq!(
            url_from_path(Path::new("/home/u/my project")).as_deref(),
            Some("file:///home/u/my%20project")
        );
        assert_eq!(
            url_from_path(Path::new("/a/b#c%d")).as_deref(),
            Some("file:///a/b%23c%25d")
        );
        // Unreserved bytes and the separator stay as they are.
        assert_eq!(
            url_from_path(Path::new("/a-b_c.d~e/f")).as_deref(),
            Some("file:///a-b_c.d~e/f")
        );
    }

    #[test]
    fn url_from_path_round_trips_through_path_from_uri() {
        for p in [
            "/etc/hosts",
            "/home/u/my project/src/main.rs",
            "/a/b#c%d",
            "/tmp/ünïcode/файл.rs",
        ] {
            let uri = url_from_path(Path::new(p)).expect("absolute path");
            assert_eq!(path_from_uri(&uri), p, "round trip of {p}");
        }
    }

    #[test]
    fn path_from_uri_leaves_a_non_file_uri_alone() {
        // jdtls, pyright and rust-analyzer all answer with URIs that are
        // not paths. Printing them unchanged is honest; stripping a
        // `file://` that isn't there and calling the rest a path is not.
        for uri in [
            "jar:file:///jdk.jar!/java/lang/String.java",
            "zipfile:///site-packages.zip/foo.py",
            "rust-analyzer://expanded/main.rs",
        ] {
            assert_eq!(path_from_uri(uri), uri);
        }
    }

    #[test]
    fn path_from_uri_keeps_a_lone_percent_verbatim() {
        // A `%` not followed by two hex digits is a filename character,
        // not a broken escape.
        assert_eq!(path_from_uri("file:///a/100%/b"), "/a/100%/b");
        assert_eq!(path_from_uri("file:///a/%zz"), "/a/%zz");
        assert_eq!(path_from_uri("file:///a/%"), "/a/%");
    }

    #[test]
    fn format_diagnostic_renders_a_percent_encoded_uri_as_a_real_path() {
        // Pins the `path_from_uri` retrofit: diagnostics now render the
        // URI a server produced, not only the one we sent.
        let d = diag(0, 0, 1, "boom", None);
        let out = format_diagnostic("file:///work/my%20dir/main.rs", &d);
        assert_eq!(out, "/work/my dir/main.rs:1:1 error: boom");
    }

    #[test]
    fn apply_cap_stops_at_the_row_limit_and_at_the_char_budget() {
        let rows: Vec<String> = (0..10).map(|i| format!("row {i}")).collect();
        assert_eq!(apply_cap(&rows, 100, 10_000), 10);
        assert_eq!(apply_cap(&rows, 3, 10_000), 3);
        // "row N" is 5 chars + 1 for the joining newline.
        assert_eq!(apply_cap(&rows, 100, 18), 3);
        // Never zero rows: one over-budget row beats an empty answer.
        let huge = vec!["x".repeat(50_000)];
        assert_eq!(apply_cap(&huge, 100, 10), 1);
    }

    #[test]
    fn file_tally_lists_the_busiest_files_first() {
        let paths: Vec<String> = ["/a", "/b", "/a", "/c", "/a", "/b", "/d"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(file_tally(&paths, 2), "/a 3, /b 2, +2 more files");
        assert_eq!(file_tally(&paths, 4), "/a 3, /b 2, /c 1, /d 1");
        // Ties break by path so the same query renders the same line —
        // `HashMap` iteration order does not.
        assert_eq!(file_tally(&paths, 3), "/a 3, /b 2, /c 1, +1 more files");
    }

    #[test]
    fn provider_enabled_reads_an_options_object_as_yes() {
        // The spec types these `boolean | XOptions`, so an options object
        // — even an empty one — is an affirmative.
        assert!(provider_enabled(&json!(true)));
        assert!(provider_enabled(&json!({})));
        assert!(provider_enabled(&json!({ "workDoneProgress": true })));
        assert!(!provider_enabled(&json!(false)));
        assert!(!provider_enabled(&Value::Null));
    }

    #[test]
    fn root_matches_everything_when_no_pattern_is_configured() {
        // An unset key has always meant "start this server anywhere", and
        // wiring the field must not quietly change that.
        assert!(root_matches(Path::new("/"), &[]));
    }

    #[test]
    fn root_matches_walks_up_from_the_launch_directory() {
        // teleia is routinely launched from a subdirectory of the
        // workspace its servers serve.
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        // Present in this crate's own directory…
        assert!(root_matches(crate_dir, &["Cargo.toml".into()]));
        // …and only at the workspace root above it, so this one can only
        // pass via the upward walk.
        assert!(root_matches(crate_dir, &[".github".into()]));
        // Any one pattern matching is enough.
        assert!(root_matches(
            crate_dir,
            &["nothing.xyz".into(), "Cargo.toml".into()]
        ));
        assert!(!root_matches(
            crate_dir,
            &["definitely-not-here.xyz".into()]
        ));
    }

    #[test]
    fn root_matches_accepts_a_glob_not_just_a_literal_name() {
        // The key is called `root_patterns`; a user writing `*.lock` and
        // getting silence would be the worst of both worlds.
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(root_matches(crate_dir, &["*.lock".into()]));
        assert!(!root_matches(crate_dir, &["*.nope-xyz".into()]));
    }

    #[test]
    fn a_registry_with_no_clients_offers_and_handles_nothing() {
        // The catalogue is conditional on a server actually running, and
        // `handles` has to agree with it: a name offered but not handled
        // routes to the built-in dispatcher and dies as "unknown tool".
        let reg = LspRegistry {
            clients: Vec::new(),
            warnings: Vec::new(),
        };
        assert!(reg.definitions().is_empty());
        for name in LSP_TOOLS {
            assert!(!reg.handles(name), "{name}");
        }
    }

    #[test]
    fn url_from_path_absolute_only() {
        assert_eq!(
            url_from_path(Path::new("/etc/hosts")).as_deref(),
            Some("file:///etc/hosts")
        );
        assert!(url_from_path(Path::new("relative/path")).is_none());
    }
}
