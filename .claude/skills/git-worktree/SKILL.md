---
name: git-worktree
description: The git workflow for football-auto-stat - one git worktree and branch per task under .claude/worktrees/, commits on that branch, merge into main from the main checkout, then clean up. Use before starting any code change, when finishing a task, when running agents in parallel, or when a git hook blocks an action.
argument-hint: "[start <task> | finish | status]"
---

# Git worktree workflow

Request: `$ARGUMENTS` (if empty: show status and say what the next step is).

The **main checkout** (`football-auto-stat/`) always stays on `main` with no code changes.
Every task gets its own **branch + worktree** under `.claude/worktrees/<task>` (gitignored).
The `git_guard.py` hook enforces this: code edits and code commits on `main` in the main checkout are blocked.
Only `todo.md`, `CLAUDE.md` and `.claude/` (planning and tooling) may be edited and committed directly on `main`.

## status

```bash
git worktree list
git -C <each worktree> status --short --branch
```
Report each worktree's branch, uncommitted changes and commits ahead of `main` (`git log --oneline main..<branch>`).

## start <task>

1. Name the task in kebab-case and choose a branch type: `feat/`, `fix/`, `refactor/`, `test/`, `docs/`, `chore/`.
2. Create the worktree – preferably with the **EnterWorktree** tool (it also moves the session into it). Otherwise:
   ```bash
   git worktree add .claude/worktrees/<task> -b <type>/<task> main
   ```
   and do all further work with paths inside `.claude/worktrees/<task>/`.
3. For agents working in parallel (detection / stats / integration), each agent gets its own worktree (`isolation: "worktree"` on the Agent tool) so they never edit the same checkout.

## While working

- Commit in the worktree in small, focused commits: `<type>: <what and why>` (e.g. `fix: use both boxes when computing distance`).
- Before committing code, run the create-tests and clean-up-code skills on the changed files and make sure `py -3.13 -m pytest -m "not slow"` passes.
- If `main` moved on, update the branch from inside the worktree: `git rebase main` (only while the branch is unpushed; otherwise `git merge main`).

## finish

1. In the worktree: everything committed (`git status` is clean), tests pass.
2. Merge into `main` (ask first only if the user hasn't already said to merge). Then, from the **main checkout**:
   ```bash
   git merge --no-ff <type>/<task>
   ```
   Resolve conflicts if any, then run the tests again on `main`.
3. Clean up:
   ```bash
   git worktree remove .claude/worktrees/<task>
   git branch -d <type>/<task>
   git worktree prune
   ```
   (or ExitWorktree with remove, when the worktree was created by EnterWorktree).
4. Push automatically (standing permission from the user): `git push origin main`, and `git push -u origin <type>/<task>` for the task branch (before deleting it) so it is on GitHub too. Never force-push.
5. Mark the task in todo.md with the manage-todo skill.

## Never

- `git checkout`/`git switch` to another branch in the main checkout – use a worktree.
- Commit code directly on `main`.
- `--no-verify`, force-push, `git reset --hard`, `git clean -f`, or `git worktree remove --force` on a worktree with uncommitted work, unless the user explicitly asks.
