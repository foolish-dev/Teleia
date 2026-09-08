# Coding guidelines (Rust)

Treat your own output as a plausible sample, not a proof: the compiler
and the tests grade it, not how right it reads.

## 1. Dumbest version first
Get the whole path compiling end to end with the dumbest body that
type-checks — a constant, a `todo!()` — then fill it in; never debug
plumbing and logic at once. Write the least code that solves the
problem: no trait, generic, lifetime or feature flag until a second
implementor or the compiler forces one. Do not add a crate for what
twenty lines of `std` do, or hand-roll what the workspace already
depends on. Keep tunables as `const` items at the top of the file that
reads them until a caller must vary one.

## 2. Look at the value, not your model of it
When behaviour surprises you, print the thing itself where it crosses a
boundary: `dbg!`, `{:#?}` over `{}`, `hexdump` the file, `diff` the two
outputs. Suspect the boring things first: a trailing `\r\n`, bytes vs
chars, the path you passed vs the one that opened.

## 3. Assume the failure is silent
Errors vanish quietly: `let _ =`, `.ok()`, `unwrap_or_default()`, an `if
let Ok(_)` with no else, a `#[cfg]`-ed-out module, a file never wired
into the mod tree, a stale binary. When an edit changes nothing, prove
the line runs: a temporary `panic!` there must actually fire. A passing
suite is no evidence your test ran — check the counts and that its name
is in the output.

## 4. One variable per cycle
Reproduce with the smallest failing input before repairing anything;
`test` runs the whole workspace, so narrow through `bash`: `cargo test
-p <crate> <name>`. Change one thing, verify, keep or revert it — two
edits in flight when the suite turns red teach you nothing. Rerun the
identical command, and hunt any result that moves on its own: `HashMap`
order, timestamps, task interleaving, a stale `target/`. Distrust a fix
that worked first try: break it, watch the expected failure return,
restore it in the same edit.

## 5. Deleting is a real change
Prefer deleting code to layering over it. When a change orphans a
function, field, variant or dependency, delete it in the same edit
rather than widening visibility or adding `#[allow(dead_code)]` — though
a field a deserializer reads is not an orphan. Comment why an odd line
is necessary, not what it does.
