---
name: code-review
description: Guidance for reviewing code the user wrote or pasted — what to check, in what order, and how to phrase findings.
---

# Code Review

When the user asks you to review, check, or critique code (in the workspace or pasted into chat):

1. Read the whole file first (use `read_file` if it's in the workspace) before commenting on any one part — a fix suggested in isolation often breaks something two lines away.
2. Check in this order: correctness bugs first, then security issues (unsanitized input, path traversal, secrets), then simplification/readability last. Don't lead with style nitpicks.
3. For each finding, state the concrete failure scenario (what input/state causes it to break), not just "this could be a problem."
4. If you can fix it directly with `edit_file`, offer to — don't just describe the fix in prose when a targeted edit is faster for the user.
5. If the code runs (a script, not a snippet), offer to actually run it with `run_python` or `run_shell` to confirm your understanding before suggesting changes based on guesswork.
