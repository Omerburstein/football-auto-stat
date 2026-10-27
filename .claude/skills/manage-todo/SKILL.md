---
name: manage-todo
description: Maintain the project's todo.md - add new tasks, keep numbering and order consistent, and mark tasks or sub-tasks as completed. Use whenever a todo.md task is finished, when new work is requested that should be tracked, or when asked to update/clean the todo list.
argument-hint: "[add <task> | done <number> | tidy]"
---

# Manage todo.md

Request: `$ARGUMENTS` (if empty: mark whatever was just finished in this session as done, then tidy).

The file is `todo.md` in the project root. Read it fully before editing.

## Format

```markdown
1. [x] Top-level task (done 2026-10-27)
    i. [x] Sub-task
    ii. [x] Sub-task
2. [ ] Top-level task
    i. [ ] Sub-task
    Notes / assumptions stay as indented plain lines without a checkbox.
```

- Top-level items: `N. [ ] text` – numbered 1, 2, 3 …
- Sub-items: 4-space indent, lowercase roman numerals `i.`, `ii.`, `iii.`, `iv.` … with their own checkbox.
- Plain descriptive lines (assumptions, notes) keep their indentation and get no checkbox.
- If the file is still in the old format (no checkboxes), convert it to this format the first time you edit it, keeping all text.

## Operations

- **Add**: append a new top-level item at the end (or as a sub-item under the task it belongs to). Keep the user's wording; fix only obvious typos. Never reorder existing numbers – other conversations refer to tasks by number.
- **Mark done**: tick `[x]` on each finished sub-item. Tick the parent and append ` (done YYYY-MM-DD)` (today's date) only when all of its sub-items are done. If only part of a task was done, tick only those sub-items and add a short indented note on what is left.
- **Tidy**: fix numbering gaps or duplicates inside sub-lists, consistent indentation, one item per line. Do not delete completed items unless the user asks – they are the history.

## Rules

- Only mark something done when it really is done (code written, and tests pass if it is code). If unsure, ask.
- Do not invent tasks; add only what the user asked for or explicitly agreed to.
- After editing, show the user the changed lines.
