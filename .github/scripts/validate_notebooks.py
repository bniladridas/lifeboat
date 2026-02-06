#!/usr/bin/env python3
"""Validate all Jupyter notebooks in the repository."""

import sys
from pathlib import Path

import nbformat


def main() -> int:
    """Validate all notebooks."""
    errors = 0
    for notebook in Path(".").rglob("*.ipynb"):
        if any(part in {".git", ".venv", "node_modules"} for part in notebook.parts):
            continue
        try:
            nbformat.read(notebook, as_version=4)
            print(f"OK: {notebook}")
        except Exception as e:
            print(f"ERROR: {notebook} - {e}")
            errors += 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
