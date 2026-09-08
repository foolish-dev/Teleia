use anyhow::{anyhow, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use teleia_llm::Message;

pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open() -> Result<Self> {
        Self::open_at(&data_path()?)
    }

    pub fn open_at(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("mkdir {parent:?}"))?;
        }
        create_owner_only(path);
        let conn = Connection::open(path).with_context(|| format!("open {path:?}"))?;
        restrict_to_owner(path);
        // A second connection to the same file (e.g. the SIGUSR1 theme
        // reload opens its own) can collide with an in-flight write.
        // Without a busy timeout SQLite returns SQLITE_BUSY immediately
        // rather than waiting, which can hard-error a turn's persistence.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                seq INTEGER NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
            CREATE TABLE IF NOT EXISTS aliases (
                name TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS input_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_input_history_time
                ON input_history(created_at DESC);",
        )?;
        Ok(Self { conn })
    }

    /// Persist a key→value preference (theme, permission_mode, notify…).
    /// Idempotent — same key overwrites.
    pub fn set_pref(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO prefs (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_pref(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT value FROM prefs WHERE key = ?1",
                params![key],
                |row| row.get::<_, String>(0),
            )
            .optional()?)
    }

    /// Append one submitted input line to the persistent readline
    /// history. Skipped when the line is empty or duplicates the most
    /// recent entry (shell-style dedup).
    pub fn push_input_history(&self, line: &str) -> Result<()> {
        if line.is_empty() {
            return Ok(());
        }
        let last: Option<String> = self
            .conn
            .query_row(
                "SELECT line FROM input_history ORDER BY id DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if last.as_deref() == Some(line) {
            return Ok(());
        }
        self.conn.execute(
            "INSERT INTO input_history (line, created_at) VALUES (?1, ?2)",
            params![line, unix_seconds()],
        )?;
        Ok(())
    }

    /// Recent input history (most recent last), capped at `limit` so an
    /// ancient runaway log doesn't blow startup memory.
    pub fn input_history(&self, limit: usize) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT line FROM (
                SELECT line, id FROM input_history ORDER BY id DESC LIMIT ?1
            ) ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![limit as i64], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn create_session(&self, model: &str) -> Result<String> {
        let id = ulid::Ulid::new().to_string();
        let now = unix_seconds();
        self.conn.execute(
            "INSERT INTO sessions (id, model, created_at) VALUES (?1, ?2, ?3)",
            params![id, model, now],
        )?;
        Ok(id)
    }

    pub fn append(&self, session_id: &str, seq: usize, message: &Message) -> Result<()> {
        let payload = serde_json::to_string(message)?;
        self.conn.execute(
            "INSERT INTO messages (session_id, seq, payload) VALUES (?1, ?2, ?3)",
            params![session_id, seq as i64, payload],
        )?;
        Ok(())
    }

    pub fn load(&self, session_id: &str) -> Result<Vec<Message>> {
        let mut stmt = self
            .conn
            .prepare("SELECT payload FROM messages WHERE session_id = ?1 ORDER BY seq ASC")?;
        let rows = stmt.query_map(params![session_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            let payload = row?;
            // Skip a single corrupt/legacy/truncated row rather than
            // aborting the whole load — one bad row must not make an
            // otherwise-valid session unresumable (`--resume` unwraps this).
            match serde_json::from_str::<Message>(&payload) {
                Ok(message) => out.push(message),
                Err(e) => eprintln!("skipping unreadable message in session {session_id}: {e}"),
            }
        }
        Ok(out)
    }

    /// The next free `seq` for a session: one past the highest stored, or 0
    /// when empty. Derived from `MAX(seq)` rather than the loaded message
    /// count so that a row skipped by [`Store::load`] (corrupt/legacy payload)
    /// can't make the next append reuse a live seq — which would collide and
    /// reorder history on the next `ORDER BY seq` load.
    pub fn next_seq(&self, session_id: &str) -> Result<usize> {
        let next: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )?;
        Ok(next as usize)
    }

    pub fn save_alias(&self, name: &str, session_id: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO aliases (name, session_id, created_at) VALUES (?1, ?2, ?3)",
            params![name, session_id, unix_seconds()],
        )?;
        Ok(())
    }

    pub fn resolve_alias(&self, name: &str) -> Result<String> {
        let id: Option<String> = self
            .conn
            .query_row(
                "SELECT session_id FROM aliases WHERE name = ?1",
                params![name],
                |row| row.get(0),
            )
            .optional()?;
        id.ok_or_else(|| anyhow!("no session saved as '{name}'"))
    }

    pub fn list_aliases(&self) -> Result<Vec<(String, String, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, session_id, created_at FROM aliases ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn delete_alias(&self, name: &str) -> Result<()> {
        let changed = self
            .conn
            .execute("DELETE FROM aliases WHERE name = ?1", params![name])?;
        if changed == 0 {
            return Err(anyhow!("no alias named '{name}'"));
        }
        Ok(())
    }
}

/// Where to keep the sqlite store. Honours `$XDG_DATA_HOME` if set
/// (works for users who symlink it on macOS too); otherwise falls
/// back to the OS-native data dir:
///
/// - Linux  : `$HOME/.local/share/teleia/teleia.sqlite`
/// - macOS  : `$HOME/Library/Application Support/teleia/teleia.sqlite`
/// - Windows: `%APPDATA%\teleia\teleia.sqlite`
/// - other  : `$HOME/.teleia/teleia.sqlite` as a last resort
fn data_path() -> Result<PathBuf> {
    if let Some(v) = std::env::var_os("XDG_DATA_HOME") {
        if !v.is_empty() {
            return Ok(PathBuf::from(v).join("teleia").join("teleia.sqlite"));
        }
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").context("HOME not set")?;
        Ok(PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("teleia")
            .join("teleia.sqlite"))
    }
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var_os("APPDATA").context("APPDATA not set")?;
        Ok(PathBuf::from(appdata).join("teleia").join("teleia.sqlite"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let home = std::env::var_os("HOME").context("HOME not set")?;
        Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("teleia")
            .join("teleia.sqlite"))
    }
}

/// Create the store file owner-only, before SQLite can create it through
/// the process umask.
///
/// [`restrict_to_owner`] repairs the mode, but a chmod cannot revoke a
/// descriptor: `Connection::open` creates the file `0644` on a stock
/// account, and another local user looping on `open(2)` can hold an fd
/// through the tightening and read every key written afterwards while
/// `ls -l` shows `0600`. Creating it ourselves closes that window — the
/// file never exists at a readable mode.
///
/// No-op when the file already exists (`create_new` fails `AlreadyExists`),
/// so the repair path below still owns databases an earlier build wrote.
#[cfg(unix)]
fn create_owner_only(path: &Path) {
    use std::os::unix::fs::OpenOptionsExt;
    let _ = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path);
}

#[cfg(not(unix))]
fn create_owner_only(_path: &Path) {}

/// Narrow the store file to owner-only when it is readable by anyone else.
///
/// This database is a secrets file: `prefs` holds every provider API key the
/// user has entered in plaintext (`main.rs`'s `pref_key_for` → `set_pref`),
/// and `messages` holds the full conversation. [`create_owner_only`] keeps a
/// *new* file out of the `022` umask's `0644`; this is the repair half, for a
/// database an earlier build already created world-readable — without it that
/// install leaks its keys for good.
///
/// SQLite gives the rollback journal the same mode as the database it belongs
/// to, so fixing the main file before the first write covers the sidecars.
/// Best-effort: a store on a filesystem with no Unix modes (a mounted FAT
/// volume, WSL DrvFs) must still open.
#[cfg(unix)]
fn restrict_to_owner(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    let mode = meta.permissions().mode();
    if mode & 0o077 != 0 {
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

#[cfg(not(unix))]
fn restrict_to_owner(_path: &Path) {}

fn unix_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use teleia_llm::Message;

    fn tmp_db() -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("teleia-test-{}-{}.sqlite", std::process::id(), n))
    }

    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn alias_save_resolve_list_delete_roundtrip() {
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());
        let store = Store::open_at(&path).unwrap();
        let session = store.create_session("test-model").unwrap();

        store.save_alias("foo", &session).unwrap();
        store.save_alias("bar", &session).unwrap();

        assert_eq!(store.resolve_alias("foo").unwrap(), session);
        assert!(store.resolve_alias("missing").is_err());

        let aliases = store.list_aliases().unwrap();
        assert_eq!(aliases.len(), 2);
        let names: Vec<_> = aliases.iter().map(|(n, _, _)| n.as_str()).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"bar"));

        store.delete_alias("foo").unwrap();
        assert!(store.resolve_alias("foo").is_err());
        assert!(store.delete_alias("foo").is_err()); // already gone
        assert_eq!(store.list_aliases().unwrap().len(), 1);
    }

    #[test]
    fn save_alias_overwrites_existing() {
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());
        let store = Store::open_at(&path).unwrap();
        let a = store.create_session("m").unwrap();
        let b = store.create_session("m").unwrap();

        store.save_alias("x", &a).unwrap();
        store.save_alias("x", &b).unwrap();

        assert_eq!(store.resolve_alias("x").unwrap(), b);
        assert_eq!(store.list_aliases().unwrap().len(), 1);
    }

    #[test]
    fn messages_persist_in_seq_order() {
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());
        let store = Store::open_at(&path).unwrap();
        let session = store.create_session("m").unwrap();

        store
            .append(
                &session,
                0,
                &Message::User {
                    content: "hi".into(),
                },
            )
            .unwrap();
        store
            .append(
                &session,
                1,
                &Message::Assistant {
                    content: Some("hello".into()),
                    tool_calls: vec![],
                },
            )
            .unwrap();

        let messages = store.load(&session).unwrap();
        assert_eq!(messages.len(), 2);
        assert!(matches!(&messages[0], Message::User { content } if content == "hi"));
        assert!(
            matches!(&messages[1], Message::Assistant { content: Some(c), .. } if c == "hello")
        );
    }

    #[test]
    fn load_skips_corrupt_row_and_keeps_the_rest() {
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());
        let store = Store::open_at(&path).unwrap();
        let session = store.create_session("m").unwrap();

        store
            .append(
                &session,
                0,
                &Message::User {
                    content: "one".into(),
                },
            )
            .unwrap();
        // A corrupt/legacy payload lands between two valid messages.
        store
            .conn
            .execute(
                "INSERT INTO messages (session_id, seq, payload) VALUES (?1, ?2, ?3)",
                params![session, 1_i64, "{not valid json"],
            )
            .unwrap();
        store
            .append(
                &session,
                2,
                &Message::User {
                    content: "two".into(),
                },
            )
            .unwrap();

        // One bad row must not sink the whole session.
        let messages = store.load(&session).unwrap();
        assert_eq!(messages.len(), 2);
        assert!(matches!(&messages[0], Message::User { content } if content == "one"));
        assert!(matches!(&messages[1], Message::User { content } if content == "two"));
    }

    #[test]
    fn next_seq_stays_past_a_skipped_corrupt_row() {
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());
        let store = Store::open_at(&path).unwrap();
        let session = store.create_session("m").unwrap();

        // Three rows at seq 0/1/2, the middle one corrupt so load() skips it.
        store
            .append(
                &session,
                0,
                &Message::User {
                    content: "a".into(),
                },
            )
            .unwrap();
        store
            .conn
            .execute(
                "INSERT INTO messages (session_id, seq, payload) VALUES (?1, ?2, ?3)",
                params![session, 1_i64, "{bad"],
            )
            .unwrap();
        store
            .append(
                &session,
                2,
                &Message::User {
                    content: "c".into(),
                },
            )
            .unwrap();

        // load() yields 2 messages, but the next seq must be 3 (past MAX),
        // not 2 (the count) — else the next append collides with seq 2.
        assert_eq!(store.load(&session).unwrap().len(), 2);
        assert_eq!(store.next_seq(&session).unwrap(), 3);

        // A fresh session with no rows starts at 0.
        let empty = store.create_session("m").unwrap();
        assert_eq!(store.next_seq(&empty).unwrap(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn store_file_is_not_readable_by_other_users() {
        use std::os::unix::fs::PermissionsExt;
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());

        let mode = |p: &PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;

        // A freshly created store holds plaintext provider keys, so it must
        // never be born group/world-readable via the default 022 umask.
        {
            let store = Store::open_at(&path).unwrap();
            store
                .set_pref("api_key:ANTHROPIC_API_KEY", "sk-secret")
                .unwrap();
        }
        assert_eq!(mode(&path), 0o600, "fresh store must be owner-only");

        // A database an earlier build already wrote at 0644 is repaired on
        // the next open, not left leaking for the life of the install.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let store = Store::open_at(&path).unwrap();
        assert_eq!(
            mode(&path),
            0o600,
            "existing world-readable store must be tightened"
        );
        assert_eq!(
            store
                .get_pref("api_key:ANTHROPIC_API_KEY")
                .unwrap()
                .as_deref(),
            Some("sk-secret"),
            "tightening must not disturb the contents"
        );
    }

    #[cfg(unix)]
    #[test]
    fn create_owner_only_makes_the_file_and_never_truncates_an_existing_one() {
        use std::os::unix::fs::PermissionsExt;
        let path = tmp_db();
        let _cleanup = Cleanup(path.clone());
        let mode = |p: &PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;

        // The mode `Store::open_at` asserts is also reachable by repairing a
        // 0644 file after the fact, so pin the creating half on its own:
        // the file has to exist at 0600 before SQLite ever sees the path,
        // because a chmod cannot revoke an fd another user already holds.
        create_owner_only(&path);
        assert_eq!(mode(&path), 0o600, "must be created owner-only");

        // And it must be `create_new`: this runs on every open, so a
        // `create(true).write(true)` here would truncate the user's whole
        // database — every session and every saved key — on the next launch.
        std::fs::write(&path, b"existing bytes").unwrap();
        create_owner_only(&path);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"existing bytes",
            "an existing store must be left completely alone"
        );
    }
}
