---
name: clean-up-code
description: Clean up Python code in football-auto-stat - keep files short, remove unused imports and redundant/dead code, keep an OOP structure, and write informative comments. Use after adding or changing code, or when asked to clean up/refactor. Optional argument - a file or folder to work on; defaults to the code changed in git.
argument-hint: "[file-or-folder]"
---

# Clean up code

Target: `$ARGUMENTS`. If no target was given, work on the Python files that changed:
`git status --porcelain -- "*.py"` plus `git diff --name-only HEAD -- "*.py"`.
If the target is a folder, work on every `.py` file in it (recursively), excluding `training/` notebooks and `weights/`.

Cleanup must **not change behaviour**. Run the tests before and after (`py -3.13 -m pytest -m "not slow"` if `tests/` exists); if they were green before, they must be green after.

## Checklist (apply in this order)

1. **Unused imports and variables**
   - Run `ruff check --select F401,F811,F841 <target>`; remove what it reports (check each one is truly unused, e.g. not re-exported from `__init__.py`).
   - Sort imports: stdlib, third-party, local – one blank line between groups.

2. **Redundant and dead code**
   - Delete commented-out code blocks (git keeps the history), empty files, unused functions/constants (search with Grep before deleting).
   - Merge duplicated logic into one helper; remove no-op statements and unreachable branches.
   - Replace magic numbers with named constants in `consts.py` (with the unit in the name or a comment).

3. **Short files**
   - Aim for ≤ 200 lines per file and ≤ 40 lines per function. When a file is longer, split it by responsibility (e.g. detection, tracking, stats, visualization) into its own module/package, and update every import.
   - `main.py` should only wire things together: parse config, build objects, run the pipeline. Move logic out of it into classes.

4. **OOP structure**
   - Domain concepts are classes: `TrackableObject` -> `Player`, `Ball`, (goals, teams, cameras as they appear). Behaviour lives on the class that owns the data, not in free functions that reach into objects.
   - Each class has one responsibility; prefer composition over deep inheritance.
   - Program to small interfaces/abstract base classes where an implementation will be swapped (detector model, tracker, camera source, stat calculator) so new variants can be added without editing callers.
   - Use type hints on public methods, `@property` for derived values, and keep attributes that callers should not touch prefixed with `_`.
   - Classes `PascalCase`, functions/variables `snake_case`, constants `UPPER_SNAKE_CASE`; module file names `snake_case.py` (rename with `git mv` and fix imports).

5. **Informative comments**
   - Every public class and method gets a short docstring: what it does, units of inputs/outputs (pixels, frames, km/h), and what it returns when data is missing.
   - Inline comments explain *why* (a threshold choice, a workaround), never restate the code.
   - Remove stale or misleading comments; fix typos in identifiers and comments.

## Finish

- Run `ruff check <target>` and the tests once more.
- Report per file: what was removed, split, renamed or documented. List anything suspicious you noticed but did not change because it would alter behaviour (likely bugs) – do not silently fix them.
