"""Claude Code hook that makes sure the project skills are used when code is added.

Modes (first CLI argument):
  post-edit   PostToolUse on Write|Edit: records changed source files and reminds Claude
              to run create-tests and clean-up-code (or manage-todo for todo.md).
  skill-used  PostToolUse on Skill: records which project skill was run.
  stop        Stop: blocks the stop once if source code changed but the required
              skills were not run afterwards.

State is kept per session in .claude/hooks/state/<session_id>.json (gitignored).
"""
import json
import sys
from pathlib import Path

REQUIRED_SKILLS = ("create-tests", "clean-up-code")
# Paths whose edits do not count as "new code" (tests are produced by the skills themselves).
IGNORED_PARTS = {"tests", ".claude", "training", "weights", "outputs"}

STATE_DIR = Path(__file__).resolve().parent / "state"
PROJECT_DIR = Path(__file__).resolve().parents[2]


def load_state(session_id: str) -> dict:
    path = STATE_DIR / f"{session_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"changed_files": [], "skills_run": []}


def save_state(session_id: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{session_id}.json").write_text(json.dumps(state), encoding="utf-8")


def clear_state(session_id: str) -> None:
    (STATE_DIR / f"{session_id}.json").unlink(missing_ok=True)


def relative_path(file_path: str) -> Path:
    """Path relative to the project, or None for files outside it (e.g. scratch scripts)."""
    try:
        return Path(file_path).resolve().relative_to(PROJECT_DIR)
    except ValueError:
        return None


def is_source_file(rel: Path) -> bool:
    return rel is not None and rel.suffix == ".py" and not IGNORED_PARTS.intersection(rel.parts)


def emit_context(event: str, text: str) -> None:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))


def on_post_edit(payload: dict) -> None:
    file_path = payload.get("tool_input", {}).get("file_path", "")
    rel = relative_path(file_path)

    if rel is not None and rel.name.lower() == "todo.md":
        emit_context("PostToolUse", "todo.md was edited: follow the manage-todo skill's format "
                                    "(checkboxes, stable numbering, mark done only when finished).")
        return
    if not is_source_file(rel):
        return

    state = load_state(payload["session_id"])
    if str(rel) not in state["changed_files"]:
        state["changed_files"].append(str(rel))
    save_state(payload["session_id"], state)

    missing = [s for s in REQUIRED_SKILLS if s not in state["skills_run"]]
    if missing:
        emit_context("PostToolUse", f"Source file {rel} changed. When this piece of work is complete, "
                                    f"run the {' and '.join(missing)} skill(s) on the changed code "
                                    f"before finishing the turn.")


def on_skill_used(payload: dict) -> None:
    skill = payload.get("tool_input", {}).get("skill", "")
    if skill not in REQUIRED_SKILLS:
        return
    state = load_state(payload["session_id"])
    if state["changed_files"] and skill not in state["skills_run"]:
        state["skills_run"].append(skill)
        save_state(payload["session_id"], state)


def on_stop(payload: dict) -> None:
    session_id = payload["session_id"]
    state = load_state(session_id)
    missing = [s for s in REQUIRED_SKILLS if s not in state["skills_run"]]

    # stop_hook_active means we already blocked once; never loop - let Claude stop.
    if not state["changed_files"] or not missing or payload.get("stop_hook_active"):
        clear_state(session_id)
        return

    files = ", ".join(state["changed_files"])
    print(json.dumps({
        "decision": "block",
        "reason": f"Source code changed in this turn ({files}) but these skills were not run: "
                  f"{', '.join(missing)}. Run them on the changed files now (use the Skill tool), "
                  f"or tell the user why they are not needed.",
    }))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.loads(sys.stdin.read() or "{}")
    payload.setdefault("session_id", "unknown")
    handlers = {"post-edit": on_post_edit, "skill-used": on_skill_used, "stop": on_stop}
    if mode in handlers:
        handlers[mode](payload)


if __name__ == "__main__":
    main()
