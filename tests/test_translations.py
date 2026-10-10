"""A translation that has stopped tracking its source is worse than none.

`docs/USAGE.zh.md` is read by somebody who cannot easily check it against
`docs/USAGE.md`. That is the whole point of having it, and it is also what makes
it dangerous: when the English changes and the Chinese does not, the Chinese
does not look stale. It looks authoritative, it is what a reader acts on, and
nothing in a normal review touches it -- the change was over in the other file.

So every `*.zh.md` records the exact content it was translated from, and this
fails when that content moves. It does **not** require the translation to be
redone in the same commit: it requires somebody to look, decide, and restamp.
The failure message says which lines moved, so deciding is cheap.

This is the same shape as `tests/test_seam.py` and the vocabulary dictionary --
where the repository keeps two copies of something, it keeps a check that they
have not drifted, because a drift here produces no error anywhere else.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: The line a translation carries, naming its source and that source's digest.
#:
#: A content digest rather than a commit: a commit touches a file for reasons
#: that have nothing to do with the prose, and a rename would break the link
#: entirely. The digest moves when and only when the words do.
MARKER = re.compile(
    r"<!--\s*tracks:\s*(?P<source>\S+?)\s*@\s*sha256:(?P<digest>[0-9a-f]{16})\s*-->"
)

#: Directories a translation may live in. Not the whole tree: `.claude/skills/`
#: is instructions an agent reads, and its `description:` decides whether the
#: skill triggers at all, so those stay in one language.
ROOTS = ("", "docs", "deploy", "notebooks")


def digest_of(path: Path) -> str:
    """The first 16 hex of the source's SHA-256. Short enough to read in a diff."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def translations() -> list[Path]:
    found: list[Path] = []
    for root in ROOTS:
        found += sorted((REPO / root).glob("*.zh.md"))
    return found


def test_there_is_at_least_one_translation_to_check() -> None:
    """The control group for every assertion below.

    Each of them iterates `translations()`, so all of them pass vacuously the
    day the glob stops matching -- a directory renamed, a suffix changed from
    `.zh.md` to `.zh-CN.md`. That failure is silent and it removes the check
    from the repository without removing the file it was checking.
    """
    assert translations(), (
        f"no *.zh.md found under {ROOTS}. If translations moved, move this glob "
        f"with them; if they were deleted, delete this file on purpose."
    )


@pytest.mark.parametrize("zh", translations(), ids=lambda p: p.name)
def test_a_translation_names_the_source_it_was_made_from(zh: Path) -> None:
    text = zh.read_text("utf-8")
    m = MARKER.search(text)
    assert m, (
        f"{zh.relative_to(REPO)} carries no `tracks:` marker, so nothing can tell "
        f"whether it is current. Put this on its first line:\n"
        f"    <!-- tracks: {zh.name.replace('.zh.md', '.md')} @ sha256:<digest> -->\n"
        f"    (python -c \"import hashlib,sys;"
        f"print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])\" <file>)"
    )
    source = (zh.parent / m.group("source")).resolve()
    assert source.is_file(), (
        f"{zh.relative_to(REPO)} tracks {m.group('source')}, which is not there. "
        f"A translation of a document that no longer exists is a document nobody "
        f"will delete."
    )


@pytest.mark.parametrize("zh", translations(), ids=lambda p: p.name)
def test_a_translation_still_tracks_what_it_was_made_from(zh: Path) -> None:
    """The one that matters. Fails when the English moved and the Chinese did not.

    Restamping is deliberate work: read what changed, decide whether the
    translation needs it, change it if so, then update the digest. Updating the
    digest without reading is the only way to defeat this, and it is at least a
    visible line in a diff.
    """
    m = MARKER.search(zh.read_text("utf-8"))
    if m is None:
        pytest.skip("no marker; the test above is the one that reports it")
    source = zh.parent / m.group("source")
    if not source.is_file():
        pytest.skip("missing source; the test above is the one that reports it")

    now = digest_of(source)
    assert m.group("digest") == now, (
        f"{source.relative_to(REPO)} has changed since "
        f"{zh.relative_to(REPO)} was written.\n"
        f"           tracked  sha256:{m.group('digest')}\n"
        f"           now      sha256:{now}\n"
        f"           Read what moved, bring the translation along if it needs it, "
        f"then restamp:\n"
        f"           python3 tools/checks/stamp_translations.py"
    )
