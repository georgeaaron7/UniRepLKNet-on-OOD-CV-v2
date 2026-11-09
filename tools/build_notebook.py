"""Build the self-contained Kaggle notebook from the readable percent-format source."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / (sys.argv[1] if len(sys.argv) > 1 else "realwaste_siglip2.py")
OUTPUT = SOURCE.with_suffix(".ipynb")

def build():
    cells = []
    current, kind = [], None

    def flush():
        if kind is None:
            return
        text = "".join(current).strip("\n") + "\n"
        if kind == "markdown":
            text = "\n".join(line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                             for line in text.splitlines()) + "\n"
        else:
            ast.parse(text)
        cell = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
                "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in SOURCE.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            current = []
        else:
            current.append(line)
    flush()
    ast.parse(SOURCE.read_text(encoding="utf-8"))
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {"accelerator": "gpu", "dataSources": [],
                   "isGpuEnabled": True, "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 5}
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Built {OUTPUT.name}: {len(cells)} cells; all code cells parse.")

if __name__ == "__main__":
    build()
