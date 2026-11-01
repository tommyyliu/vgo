#!/usr/bin/env python3
"""Check that cited lemmas exist and that links in maintained docs resolve.

Files under docs/history are dated records, so their links are not checked.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITATION = re.compile(r"\[([AENO]\d{1,2})\]")
HEADING = re.compile(r"^#{2,3} ([AENO]\d{1,2})\.", re.M)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def tracked(*patterns: str) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", *patterns], cwd=ROOT, text=True)
    return [ROOT / line for line in output.split()]


def main() -> int:
    lemmas = (ROOT / "spec/LEMMAS.md").read_text()
    known = set(HEADING.findall(lemmas))
    problems = []
    used = set()
    for path in tracked("*.rs", "spec/*.md"):
        for lemma in CITATION.findall(path.read_text()):
            used.add(lemma)
            if lemma not in known:
                problems.append(f"{path.relative_to(ROOT)}: cites unknown [{lemma}]")
    for lemma in sorted(k for k in known if k[0] in "EN" and k not in used):
        problems.append(f"{lemma} is cited by no code")
    docs = [p for p in tracked("*.md") if "docs/history/" not in str(p)]
    for path in docs:
        for target in LINK.findall(path.read_text()):
            if re.match(r"[a-z]+:|#", target):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                problems.append(f"{path.relative_to(ROOT)}: broken link {target}")
    print("\n".join(problems) if problems else f"spec ok: {len(known)} lemmas, {len(docs)} docs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
