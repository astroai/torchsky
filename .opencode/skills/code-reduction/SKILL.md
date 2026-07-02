---
name: code-reduction
description: Remove bloat, shorten diffs, prefer stdlib, delete dead code — ponytail discipline
license: MIT
compatibility: opencode
metadata:
  audience: all
  principle: yagni
---

## The ladder (stop at the first rung that holds)

1. Does this need to exist? Speculative = skip it.
2. Already in codebase? Reuse before rewrite.
3. Stdlib does it? Use it.
4. Native platform feature covers it? CSS > JS, DB constraint > app code.
5. Already-installed dependency solves it? Never add one for what a few lines can do.
6. One line? One line.

## Rules

- No interface with one implementation, no factory for one product, no config for a value that never changes.
- Deletion over addition. Shortest working diff wins.
- Fewest files possible. Boring over clever.
- Mark deliberate simplifications with `ponytail:` and name the ceiling + upgrade path.

## Check before adding code

- Can I delete something instead?
- Is there an existing function that almost does this?
- Am I solving a real bottleneck or anticipating one?
