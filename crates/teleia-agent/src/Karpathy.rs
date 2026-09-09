//! Behavioral guidelines in the spirit of Andrej Karpathy's public
//! writing on code and debugging. The agent appends [`GUIDELINES`] to
//! its system prompt so the model inherits the same discipline.
//!
//! Vendored verbatim from `CLAUDE.md` in
//! <https://github.com/multica-ai/andrej-karpathy-skills>, so the advice
//! is language-agnostic rather than written against `cargo` and this
//! workspace. Re-fetch the upstream file to update it; do not edit the
//! local copy, or the next re-fetch silently discards the change.
//!
//! Sourced from `Karpathy.md` at the workspace root so the same file is
//! shared between the agent prompt and the human-facing reference. Edit
//! the .md and rebuild — both stay in sync.

/// The principles, formatted for direct inclusion in a system
/// prompt. Self-contained — safe to concatenate after the base prompt
/// with a single blank line.
pub const GUIDELINES: &str = include_str!("../../../Karpathy.md");
