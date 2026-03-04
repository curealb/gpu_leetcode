import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python run.py <exercise_dir>")

    repo_root = Path(__file__).resolve().parent
    exercise_dir = (repo_root / sys.argv[1]).resolve()
    script_path = exercise_dir / "main.py"

    if not exercise_dir.is_dir():
        raise SystemExit(f"Exercise directory not found: {sys.argv[1]}")

    if not script_path.is_file():
        raise SystemExit(f"main.py not found in: {sys.argv[1]}")

    # Keep shared imports resolvable while preserving per-exercise relative file access.
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(exercise_dir))

    previous_cwd = Path.cwd()
    try:
        import os

        os.chdir(exercise_dir)
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()