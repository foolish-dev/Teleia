//! Behavioral guidelines in the spirit of Andrej Karpathy's public
//! writing on code and debugging, rewritten for a Rust workspace. The
//! agent appends [`GUIDELINES`] to its system prompt so the model
//! inherits the same discipline — but with examples that fit `cargo`,
//! traits, and a shell-tool loop rather than generic LLM-coding advice.
//!
//! Sourced from `Karpathy.md` at the workspace root so the same file is
//! shared between the agent prompt and the human-facing reference. Edit
//! the .md and rebuild — both stay in sync.

/// The principles, formatted for direct inclusion in a system
/// prompt. Self-contained — safe to concatenate after the base prompt
/// with a single blank line.
pub const GUIDELINES: &str = include_str!("../../../Karpathy.md");
