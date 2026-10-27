"""Claude Code hook that enforces the git worktree workflow (see the git-worktree skill).

The main checkout stays on main and clean; every code change happens on a branch in its
own worktree under .claude/worktrees/, and reaches main only through a merge.

Modes (first CLI argument):
  session-start  SessionStart: tells Claude where it is (main checkout or worktree).
  pre-edit       PreToolUse on Write|Edit: blocks source edits in the main checkout on main.
  pre-bash       PreToolUse on Bash: blocks/asks for git commands that break the workflow.
"""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROTECTED_BRANCHES = {"main", "master"}
WORKTREES_DIR = Path(".claude") / "worktrees"
# Planning/tooling files that may be edited and committed directly on main.
EXEMPT_PATHS = ("todo.md", "CLAUDE.md", ".claude/")
WORKFLOW_HINT = ("Follow the git-worktree skill: start a worktree with the EnterWorktree tool "
                 "(or `git worktree add .claude/worktrees/<task> -b <type>/<task>`), work and "
                 "commit there, then merge into main from the main checkout.")


class GitContext:
    """Where a path lives in git: repo root, current branch, and main checkout vs. linked worktree."""

    def __init__(self, root: Path, branch: str, is_linked_worktree: bool):
        self.root = root
        self.branch = branch
        self.is_linked_worktree = is_linked_worktree

    @property
    def is_protected_main_checkout(self) -> bool:
        return not self.is_linked_worktree and self.branch in PROTECTED_BRANCHES

    @classmethod
    def of(cls, path: Path) -> Optional["GitContext"]:
        """Returns None when path is not inside a git repository."""
        path = path if path.is_dir() else path.parent
        while not path.exists() and path != path.parent:  # new files in new folders
            path = path.parent
        out = run_git(path, "rev-parse", "--path-format=absolute", "--show-toplevel",
                      "--git-dir", "--git-common-dir", "--abbrev-ref", "HEAD")
        if out is None:
            return None
        root, git_dir, common_dir, branch = out.splitlines()
        return cls(Path(root), branch, Path(git_dir).resolve() != Path(common_dir).resolve())

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def run_git(cwd: Path, *args: str) -> Optional[str]:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def is_exempt(rel_path: str) -> bool:
    return any(rel_path == p or (p.endswith("/") and rel_path.startswith(p)) for p in EXEMPT_PATHS)


def decide(event: str, decision: str, reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "permissionDecision": decision,
                                             "permissionDecisionReason": reason}}))


# ---------- session-start ----------

def on_session_start(payload: dict) -> None:
    ctx = GitContext.of(Path(payload.get("cwd", ".")))
    if ctx is None:
        return
    if ctx.is_linked_worktree:
        text = (f"You are in a git worktree on branch '{ctx.branch}'. Commit your work here; "
                f"merge into main from the main checkout when the task is done (git-worktree skill).")
    else:
        worktrees = run_git(ctx.root, "worktree", "list") or ""
        text = (f"You are in the MAIN checkout on '{ctx.branch}'. Source edits here are blocked. "
                f"{WORKFLOW_HINT}\nExisting worktrees:\n{worktrees}")
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}))


# ---------- pre-edit ----------

def on_pre_edit(payload: dict) -> None:
    file_path = Path(payload.get("tool_input", {}).get("file_path", ""))
    ctx = GitContext.of(file_path)
    if ctx is None or not ctx.is_protected_main_checkout or is_exempt(ctx.relative(file_path)):
        return
    decide("PreToolUse", "deny", f"{ctx.relative(file_path)} is in the main checkout on "
                                 f"'{ctx.branch}'. Code changes go in a worktree. {WORKFLOW_HINT}")


# ---------- pre-bash ----------

def split_segments(command: str) -> list:
    """Splits a shell command into simple commands on ;, &&, ||, | and newlines."""
    segments = []
    for part in re.split(r"&&|\|\||[;|\n]", command):
        try:
            tokens = shlex.split(part, posix=True)
        except ValueError:
            tokens = part.split()
        if tokens:
            segments.append(tokens)
    return segments


def parse_git(tokens: list, cwd: Path):
    """Returns (cwd, subcommand, args) for a git command, or None. Honours `git -C <path>`."""
    if Path(tokens[0]).name.lower() not in ("git", "git.exe"):
        return None
    i = 1
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] == "-C" and i + 1 < len(tokens):
            cwd = (cwd / tokens[i + 1]).resolve()
            i += 1
        elif tokens[i] == "-c":
            i += 1
        i += 1
    if i >= len(tokens):
        return None
    return cwd, tokens[i], tokens[i + 1:]


def check_git(sub: str, args: list, ctx: GitContext, cwd: Path, full_command: str):
    """Returns (decision, reason) for one git command, or None to allow it."""
    flags = set(args)
    main_checkout = ctx.is_protected_main_checkout

    if "--no-verify" in flags or ("-n" in flags and sub == "commit"):
        return "deny", "Do not skip git hooks (--no-verify). Fix what the hook reports instead."

    if sub in ("checkout", "switch") and not ctx.is_linked_worktree:
        if "--" in args or "." in args:
            return "ask", "This discards uncommitted changes in the main checkout."
        return "deny", f"Do not switch or create branches in the main checkout. {WORKFLOW_HINT}"

    if sub == "commit" and main_checkout:
        staged = (run_git(ctx.root, "diff", "--cached", "--name-only") or "").splitlines()
        stages_more = "git add" in full_command or flags & {"-a", "--all"}
        if not stages_more and staged and all(is_exempt(p) for p in staged):
            return None  # planning/tooling files only (todo.md, .claude/)
        if stages_more:
            return "ask", (f"Committing directly on '{ctx.branch}' in the main checkout. Only todo.md, "
                           f"CLAUDE.md and .claude/ belong here; code goes through a worktree branch.")
        return "deny", f"Commits with code changes do not go directly on '{ctx.branch}'. {WORKFLOW_HINT}"

    if sub == "worktree" and args[:1] == ["add"]:
        paths = [a for a in args[1:] if not a.startswith("-")]
        for opt in ("-b", "-B"):  # the value after -b/-B is the branch name, not the path
            if opt in args and args.index(opt) + 1 < len(args):
                paths.remove(args[args.index(opt) + 1])
        target = (cwd / paths[0]).resolve() if paths else None
        inside_repo = target is not None and ctx.root.resolve() in target.parents
        if inside_repo and (ctx.root / WORKTREES_DIR).resolve() not in target.parents:
            return "deny", f"Put worktrees under {WORKTREES_DIR.as_posix()}/<task> (gitignored), not inside the source tree."
        return None

    if sub == "worktree" and args[:1] == ["remove"] and flags & {"-f", "--force"}:
        return "ask", "Force-removing a worktree deletes its uncommitted work. Commit or check `git -C <path> status` first."

    if sub == "push":
        refspecs = [a for a in args if not a.startswith("-")][1:]  # after the remote name
        if flags & {"-f", "--force", "--force-with-lease"} or any(r.startswith("+") for r in refspecs):
            return "ask", "Force push rewrites remote history."
        targets = [r.lstrip("+").split(":")[-1] for r in refspecs] or [ctx.branch]
        if PROTECTED_BRANCHES.intersection(targets):
            return "ask", "Pushing to main publishes to GitHub."

    destructive = {
        "reset": "--hard" in flags,
        "clean": any(a.startswith("-") and "f" in a for a in args),
        "restore": "--staged" not in flags,
        "branch": "-D" in flags or bool(flags & {"-d", "--delete"} and flags & {"-f", "--force"}),
        "stash": args[:1] in (["drop"], ["clear"]),
    }
    if destructive.get(sub):
        return "ask", f"`git {sub} {' '.join(args)}` permanently discards work."
    return None


def on_pre_bash(payload: dict) -> None:
    command = payload.get("tool_input", {}).get("command", "")
    if "git" not in command:
        return
    cwd = Path(payload.get("cwd", "."))
    worst = None
    for tokens in split_segments(command):
        if tokens[0] == "cd" and len(tokens) > 1:
            cwd = (cwd / tokens[1]).resolve()
            continue
        parsed = parse_git(tokens, cwd)
        if parsed is None:
            continue
        git_cwd, sub, args = parsed
        ctx = GitContext.of(git_cwd)
        verdict = check_git(sub, args, ctx, git_cwd, command) if ctx else None
        if verdict and (worst is None or verdict[0] == "deny"):
            worst = verdict
    if worst:
        decide("PreToolUse", *worst)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.loads(sys.stdin.read() or "{}")
    handlers = {"session-start": on_session_start, "pre-edit": on_pre_edit, "pre-bash": on_pre_bash}
    if mode in handlers:
        handlers[mode](payload)


if __name__ == "__main__":
    main()
