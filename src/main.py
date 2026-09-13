"""CLI entrypoint. Launches the Textual TUI (src/tui.py)."""

import sys

from src.tui import run


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run(root)


if __name__ == "__main__":
    main()
