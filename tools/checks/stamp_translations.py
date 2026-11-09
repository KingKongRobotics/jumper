#!/usr/bin/env python3
"""Stamp each `*.zh.md` with the digest of the source it tracks.

    python3 tools/checks/stamp_translations.py            # show what has moved
    python3 tools/checks/stamp_translations.py --write    # restamp

`tests/test_translations.py` fails when a translation's source has changed
since the translation was written. This is how you clear that failure **after**
bringing the translation along -- not instead of it. Run it with no arguments
first: it prints which sources moved, so you know what to read.

Separate from the test on purpose. A test that repaired itself would report
nothing, and the report is the useful part.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARKER = re.compile(
    r"(?P<head><!--\s*tracks:\s*)(?P<source>\S+?)(?P<mid>\s*@\s*sha256:)"
    r"(?P<digest>[0-9a-f]{16})(?P<tail>\s*-->)"
)
ROOTS = ("", "docs", "deploy", "notebooks")


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def recent_commits(source: Path) -> str:
    """The last few commits that touched the source, if git can say.

    Not a diff against the stamped digest: the digest is a content hash, not a
    revision, so there is nothing to diff it against. These are the commits to
    read.
    """
    log = subprocess.run(
        ["git", "log", "--oneline", "-3", "--", str(source.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    return log.stdout.strip() or f"(no git history for {source.name})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="restamp; without it, only report")
    args = ap.parse_args()

    found = [p for root in ROOTS for p in sorted((REPO / root).glob("*.zh.md"))]
    if not found:
        print("[stamp] no *.zh.md anywhere in", ROOTS)
        return 1

    stale = 0
    for zh in found:
        text = zh.read_text("utf-8")
        m = MARKER.search(text)
        rel = zh.relative_to(REPO)
        if not m:
            print(f"[stamp] {rel}: no `tracks:` marker")
            stale += 1
            continue
        source = zh.parent / m.group("source")
        if not source.is_file():
            print(f"[stamp] {rel}: tracks {m.group('source')}, which is not there")
            stale += 1
            continue
        now = digest_of(source)
        if now == m.group("digest"):
            continue

        stale += 1
        print(f"[stamp] {rel}")
        print(f"          {source.relative_to(REPO)} moved: "
              f"{m.group('digest')} -> {now}")
        for line in recent_commits(source).splitlines():
            print(f"          {line}")
        if args.write:
            zh.write_text(MARKER.sub(
                lambda mm, d=now: mm.group("head") + mm.group("source")
                + mm.group("mid") + d + mm.group("tail"), text, count=1), "utf-8")
            print("          restamped")

    if not stale:
        print(f"[stamp] {len(found)} translation(s), all current")
        return 0
    if not args.write:
        print("\n[stamp] read what moved, bring the translations along, "
              "then re-run with --write")
    return 0 if args.write else 1


if __name__ == "__main__":
    sys.exit(main())
