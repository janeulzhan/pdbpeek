"""Regenerate the committed summary golden file.

Run this when the template in pdbpeek/summary.py changes on purpose, then
read the diff before committing. If the diff is a surprise, the change was
not on purpose.

    python3 tools/regen_golden.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pdbpeek.summary import summarize_entry  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures")

TARGETS = [("entry_1CRN.json", "summary_1CRN.golden.txt")]


def main() -> None:
    for source, destination in TARGETS:
        with open(os.path.join(FIXTURES, source), "r", encoding="utf-8") as fh:
            entry = json.load(fh)
        text = summarize_entry(entry)
        path = os.path.join(FIXTURES, destination)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {destination} ({len(text)} chars)")
        print(f"  {text}")


if __name__ == "__main__":
    main()
