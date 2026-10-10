"""Directory structure invariants.

These assertions look trivial, but each corresponds to something that really
happened and that **raises nothing while producing wrong results**. Pure filesystem
checks, importing no simulation dependencies, runnable anywhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: This repository's own code. The vendored copies (rl/mjlab, rl/rsl_rl) stay
#: verbatim upstream and are exempt from these rules -- but rl/mjrl, also under
#: rl/, is ours and is covered.
FIRST_PARTY_DIRS = ("rl/mjrl", "tasks", "scripts", "deploy", "tools", "assets", "tests")

#: Directories that sit inside first-party trees without being first-party code.
#: `deploy/convert/` carries its own virtualenv -- rknn-toolkit2 pins numpy,
#: torch and onnx below what training needs, so it cannot share the repository's.
#: Walking it would be slow and, worse, would put ~30000 site-packages filenames
#: into the set the README tree is checked against, where they would satisfy
#: almost any token and quietly make that assertion vacuous.
NOT_OURS = frozenset({"__pycache__", ".venv"})


def _ours(path: Path) -> bool:
    return not NOT_OURS & set(path.parts)


def first_party_py() -> list[Path]:
    return sorted(
        p for d in FIRST_PARTY_DIRS for p in (REPO / d).rglob("*.py") if _ours(p)
    )


def test_rl_is_a_package_root_not_a_package() -> None:
    """`rl/` is a package root and must itself **not** be a package (no __init__.py).

    With an __init__.py, `mjlab` becomes a submodule of `rl.mjlab` and the hundreds
    of `import mjlab.envs` statements in upstream code all break. pyproject's
    `where = [".", "rl"]` publishes the three directories beneath it as top-level
    packages precisely because `rl/` is not a package.
    """
    assert (REPO / "rl").is_dir()
    assert not (REPO / "rl" / "__init__.py").exists(), "rl/ must not have an __init__.py"
    for name in ("mjrl", "mjlab", "rsl_rl"):
        assert (REPO / "rl" / name / "__init__.py").exists(), f"rl/{name} should be a package"
        assert not (REPO / name).exists(), f"{name}/ must not be at the repository root"


def test_ruff_excludes_only_vendored() -> None:
    """ruff's exclude must not be the whole of `rl` -- that would exempt our own
    rl/mjrl along with it."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    excluded = re.search(r"^exclude = (\[.*\])$", pyproject, re.MULTILINE)
    assert excluded, "no ruff exclude found in pyproject.toml"
    assert "rl/mjrl" not in excluded.group(1)
    assert '"rl"' not in excluded.group(1), (
        "exclude covers the whole of rl/, which would skip rl/mjrl"
    )


def test_no_sys_path_mutation() -> None:
    """First-party code must not rewrite sys.path.

    There were once 12 `sys.path.insert(REPO_ROOT)` calls, which made "which
    directory you ran from" an implicit dependency: from the repository root you got
    the in-repo mjlab, from anywhere else the pip version, with inconsistent
    behaviour that was very hard to notice. Everything now relies on
    `pip install -e .`.
    """
    offenders = []
    for path in first_party_py():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            # Match sys.path.insert(...) / sys.path.append(...)
            target = node.func.value
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "path"
                and isinstance(target.value, ast.Name)
                and target.value.id == "sys"
            ):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not offenders, "these still rewrite sys.path:\n  " + "\n  ".join(offenders)


def test_the_deploy_skill_quotes_a_message_the_bundler_still_produces() -> None:
    """The skill tells you not to check the two halves by reading, because the
    bundler does it. Then it quotes what the bundler says.

    A quoted message is a copy, and this one is load-bearing: somebody reading
    the skill is being told to recognise that failure and change one of two
    named files. If the wording drifts, the instruction points at a message that
    never appears. The check lives in `operator::check_switches` since
    2026-09-29, which `Bundle::open` calls; it was in `bundle.rs` before.
    """
    skill = (REPO / ".claude/skills/deploy/SKILL.md").read_text("utf-8")
    source = (REPO / "deploy/fsm/src/operator.rs").read_text("utf-8")

    # Rust lets a string literal span lines with a trailing backslash, which
    # eats the newline *and* the indentation after it. Undo that before
    # flattening, or every fragment that crosses a line looks missing.
    import re

    flat = " ".join(re.sub(r"\\\s*\n\s*", "", source).split())
    for fragment in (
        "switches on the pad's",
        "controls use too: one press would do both.",
        "Pick another control, or say with `from` that the switch is not pressed in",
    ):
        assert fragment in flat, f"the bundler no longer says {fragment!r}"
        assert fragment in " ".join(skill.split()), f"the skill no longer quotes {fragment!r}"


def test_the_export_gate_matches_what_the_controller_can_build() -> None:
    """`export.py` refuses a term the robot cannot build. Against the real list.

    The gate is a copy -- `export.py` runs on a training machine and cannot ask
    a Rust crate -- and the copy had drifted the permissive way: it claimed ten
    terms `obs.rs` has no case for, five of them the whole of what makes
    `jumper.dance` a dance. So the export would have succeeded, the bundle would
    have looked fine, and the robot would have refused it on the bench, which
    is the exact failure the gate's own docstring says it exists to prevent.

    A gate that waves things through is worse than no gate, because it is
    believed.
    """
    import re

    terms = re.search(
        r"_DEPLOY_TERMS = frozenset\(\{(.*?)\}\)",
        (REPO / "scripts/export.py").read_text("utf-8"),
        re.S,
    )
    assert terms, "_DEPLOY_TERMS is not in export.py in the shape this test reads"
    claimed = set(re.findall(r'"([^"]+)"', terms.group(1)))

    supported = re.search(
        r"const SUPPORTED: &\[&str\] = &\[(.*?)\];",
        (REPO / "deploy/fsm/src/obs.rs").read_text("utf-8"),
        re.S,
    )
    assert supported, "SUPPORTED is not in obs.rs in the shape this test reads"
    builds = set(re.findall(r'"([^"]+)"', supported.group(1)))

    # The three `layout.rs` rewrites onto `commands` before the builder sees
    # them. Read from that file too, so adding a fourth alias does not need
    # this test edited -- only the export list.
    rewrite = re.search(
        r'if matches!\(term\.name\.as_str\(\), ([^)]*)\)',
        (REPO / "deploy/fsm/src/layout.rs").read_text("utf-8"),
    )
    assert rewrite, "the alias table is not in layout.rs in the shape this test reads"
    aliases = set(re.findall(r'"([^"]+)"', rewrite.group(1)))

    assert claimed == builds | aliases, (
        f"only in export.py: {sorted(claimed - builds - aliases)}\n"
        f"only in the crate: {sorted(builds - claimed)}"
    )
    # Control group: the three parses found something, so the equality above
    # cannot hold by comparing empty sets.
    assert len(builds) >= 10 and len(aliases) == 3


def test_no_absolute_machine_paths() -> None:
    """No absolute path from one particular machine.

    Two scripts once hardcoded a developer's home directory, which meant an
    ImportError or FileNotFoundError on any other machine.
    """
    pattern = re.compile(r"""["'](/home/|/Users/)[^"']*["']""")
    offenders = []
    for path in first_party_py():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these hardcode a machine-specific absolute path:\n  " + "\n  ".join(offenders)
    )


def test_asset_scripts_live_under_their_asset_tools_dir() -> None:
    """Scripts that generate or calibrate an asset live in that asset's own `tools/`.

    The convention is in the README's "Where does a new file go": a script bound to
    a specific asset belongs in `assets/<name>/tools/` and lives and dies with it,
    while the top-level `tools/` holds only asset-independent checks and benchmarks.
    The asset directory itself holds data only.
    """
    jumper = REPO / "assets" / "jumper"
    assert (jumper / "jumper.xml").exists()
    for script in ("build_jumper.py", "check_stance.py", "check_model_slim.py"):
        assert (jumper / "tools" / script).exists(), f"assets/jumper/tools/{script} is gone"
    assert not (REPO / "tools" / "assets").exists(), "the top-level tools/assets is retired"

    strays = [p.name for p in jumper.glob("*.py")]
    assert not strays, (
        f"scripts must not sit directly in assets/jumper/; move them into tools/: {strays}"
    )


def test_top_level_tools_are_not_bound_to_one_asset() -> None:
    """A script in the top-level `tools/` must not be bound to one asset.

    The criterion is whether it imports a hardcoded asset path constant (something
    like `JUMPER_XML`). A general tool taking a model path on the command line
    (`bench_phase0.py`) does not count -- it runs on a different robot unchanged --
    whereas `check_model_slim.py` pinned itself to jumper through
    `from tasks.jumper.constants import JUMPER_XML`, which is why it belongs in
    `assets/jumper/tools/`.
    """
    offenders = []
    for path in (REPO / "tools").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name.endswith("_XML"):
                        offenders.append(
                            f"{path.relative_to(REPO)}:{node.lineno} imports {alias.name}"
                        )
    assert not offenders, (
        "these tools/ scripts are bound to a specific asset and belong in "
        "assets/<name>/tools/:\n  "
        + "\n  ".join(offenders)
    )


def test_contributing_directory_tree_matches_disk() -> None:
    """Every path mentioned in CONTRIBUTING's directory tree must really exist.

    The project overview once claimed files that were never written -- `mjrl/envs/`,
    `mjrl/config.py`, `scripts/parity.py` -- and even told readers to run
    `scripts/parity.py`.
    """
    contributing = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    tree = contributing.split("## Directory layout", 1)[1].split("```", 2)[1]

    # Collect the repository's paths first. The vendored rl/ is scanned one level
    # deep -- the tree mentions only its two subdirectories, and there is no reason
    # to walk 34 MB of STL for that.
    paths = {
        p.relative_to(REPO).as_posix()
        for d in (*FIRST_PARTY_DIRS, "docs")
        for p in (REPO / d).rglob("*")
        if _ours(p)
    }
    paths |= {p.name for p in REPO.iterdir()}
    paths |= {p.relative_to(REPO).as_posix() for p in (REPO / "rl").iterdir()}
    names = {p.rsplit("/", 1)[-1] for p in paths}

    def on_disk(token: str) -> bool:
        # The tree contains both bare filenames (`resolve.py`) and fragments with
        # levels (`smoke/smoke_biped.xml`)
        if "/" not in token:
            return token in names
        return any(p == token or p.endswith("/" + token) for p in paths)

    missing = []
    for raw in tree.splitlines():
        # Strip the tree lines and trailing comments, leaving the path fragment
        line = re.sub(r"^[│├└─\s]*", "", raw).split("#")[0].strip()
        if not line or line.startswith(("jumper/", "mjrl-lab/")):
            continue
        for token in line.split():  # one line may list several files
            name = token.rstrip("/")
            if name and not on_disk(name):
                missing.append(name)
    assert not missing, (
        "mentioned in the CONTRIBUTING tree but absent from disk: "
        + ", ".join(sorted(set(missing)))
    )


def test_usage_doc_matches_the_code() -> None:
    """What `docs/USAGE.md` promises must really exist.

    A manual goes stale most easily of all: rename a function or move a file and the
    document stays as it was, so a reader copying from it hits an error. This checks
    the manual's **key identifiers** against the code.
    """
    usage = (REPO / "docs" / "USAGE.md").read_text(encoding="utf-8")

    # The three files and two functions the manual tells people to write
    registry = (REPO / "tasks" / "registry.py").read_text(encoding="utf-8")
    for name in ("def register(", "def load_env_cfg(", "def load_agent_cfg("):
        assert name in registry, f"{name}, mentioned in USAGE, is not in the registry"

    # The .env keys in the manual must match the repository's .env
    import re

    doc_keys = set(re.findall(r"^(MJRL_[A-Z_]+)=", usage, re.M))
    env_keys = set(
        re.findall(r"^(MJRL_[A-Z_]+)=", (REPO / ".env").read_text(encoding="utf-8"), re.M)
    )
    assert doc_keys == env_keys, (
        f"USAGE and .env keys disagree.\nExtra in the document: "
        f"{sorted(doc_keys - env_keys)}\n"
        f"Missing from the document: {sorted(env_keys - doc_keys)}"
    )

    # The modules and command-line switches the manual names
    assert (REPO / "tasks" / "jumper" / "common" / "assets.py").exists()
    assert (REPO / "tasks" / "jumper" / "common" / "ppo.py").exists()
    train = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    # `--viewer-all-envs` is gone: the live viewer is on by default and draws as
    # many environments as it can, up to a cap (see _resolve_draw_limit in
    # mjrl/viewer/live.py). `--headless` turns it off.
    for flag in (
        "--headless", "--logger", "--max-iterations", "--dry-run", "--resume", "--checkpoint"
    ):
        assert flag in train, f"{flag}, mentioned in USAGE, is not in train.py"


def test_usage_doc_logger_default_claim() -> None:
    """The manual says `--logger` defaults to tensorboard -- for a reason, and it
    must not be changed back.

    mjlab defaults to wandb, and the start_method='thread' it passes to wandb is
    rejected by wandb 0.29.0's config validation, crashing training right after the
    model is built.
    """
    train = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    block = train.split('"--logger"', 1)[1].split(")", 1)[0]
    assert 'default="tensorboard"' in block


def test_agent_instructions_point_at_files_that_exist() -> None:
    """Every repository path CLAUDE.md and AGENTS.md link to must be real.

    These two are read at the start of a session and acted on without checking,
    which is what makes a stale link expensive: an agent told to consult
    `docs/DESIGN.md` after a rename does not conclude the document moved, it
    concludes the instruction was wrong and stops trusting the file. Renames are
    exactly what this repository has been doing, so the links are pinned.
    """
    import re

    for name in ("CLAUDE.md", "AGENTS.md"):
        doc = REPO / name
        assert doc.is_file(), f"{name} is missing"
        text = doc.read_text(encoding="utf-8")
        targets = re.findall(r"\]\(([^)#]+)\)", text)
        repo_paths = [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]
        assert repo_paths, f"{name} links to nothing in the repository"
        missing = [t for t in repo_paths if not (REPO / t).exists()]
        assert not missing, f"{name} links to paths that do not exist: {missing}"


def test_every_skill_is_indexed_for_agents_that_do_not_load_skills() -> None:
    """AGENTS.md links every `.claude/skills/<name>/SKILL.md`.

    Claude Code finds a skill in that directory by itself; every other agent
    finds it only through the table in AGENTS.md. A skill added without a row
    there exists for one agent and not for the rest, and nothing would say so:
    the directory is right, the skill loads, and the other agent never reads
    it. A row for a skill that is gone is the link check's job, above.
    """
    import re

    skills = sorted(p.parent.name for p in (REPO / ".claude" / "skills").glob("*/SKILL.md"))
    # The control group: an empty glob would make the assertion below pass for
    # an index of nothing, which is exactly the day the skills move.
    assert skills, "no .claude/skills/*/SKILL.md found; if the skills moved, move this glob"
    text = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    indexed = set(re.findall(r"\]\(\.claude/skills/([^/)]+)/SKILL\.md\)", text))
    missing = [s for s in skills if s not in indexed]
    assert not missing, (
        f"AGENTS.md does not index {missing}. Add a row to its skill table linking "
        f".claude/skills/<name>/SKILL.md, or agents other than Claude never find it"
    )


def test_agent_instructions_do_not_restate_the_reference_docs() -> None:
    """CLAUDE.md is loaded into every session, so length is a running cost.

    Its job is the part the reference documents assume you already know, not a
    summary of them -- a summary is what drifts, and it drifts silently because
    nothing reads both at once. Keeping it short is the mechanism that keeps it
    honest.
    """
    lines = (REPO / "CLAUDE.md").read_text(encoding="utf-8").splitlines()
    assert len(lines) < 200, (
        f"CLAUDE.md is {len(lines)} lines; it is loaded every session and the "
        f"detail belongs in docs/, which it links to"
    )


def test_plays_teardown_does_not_depend_on_how_far_startup_got() -> None:
    """`finally` must not read a name bound later in the `try`.

    This shipped and hid a message. `--bundle` refuses a contract it cannot
    build an observation from, which is a clean `SystemExit` with the reason in
    it -- and the teardown then raised `UnboundLocalError: monitor`, because the
    monitor is built after the policy and the `finally` reads it unconditionally.
    What the user saw was the second exception, with the first one folded into a
    "during handling of the above" they had to scroll past.

    Any failure between entering the `try` and building the monitor does this;
    the bundle refusal only made it reachable. So what is checked is the
    property: every name the teardown touches is bound before the `try` it
    guards.
    """
    import ast

    source = (REPO / "scripts" / "play.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="play.py")

    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "play.py has no try/finally; this test is guarding nothing"

    for node in tries:
        read: set[str] = set()
        for stmt in node.finalbody:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    read.add(sub.id)
        # Names the `try` body binds are exactly the ones that may be unbound
        # when the finally runs.
        bound_inside: set[str] = set()
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    bound_inside.add(sub.id)
        # ...unless they are also bound before it, which is the fix.
        before = set()
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.Name) and isinstance(stmt.ctx, ast.Store):
                if stmt.lineno < node.lineno:
                    before.add(stmt.id)
        risky = sorted((read & bound_inside) - before)
        assert not risky, (
            f"play.py:{node.lineno}'s finally reads {risky}, which the try body "
            f"binds -- an exception before that line replaces the real error with "
            f"UnboundLocalError"
        )


def test_the_replay_recorder_is_the_only_mp4_writer() -> None:
    """One recorder: `play.py --video`, through `rl/mjrl/viewer/video.py`.

    There were three. The README clips, the dance export and the replay each
    drew their own frames, with their own camera, their own idea of which geom
    groups to hide and their own encoder settings -- and a difference between two
    pictures of the same policy read as a difference between two policies. Now
    everything that wants an MP4 runs `play --video`; `tools/readme_media.py`
    and `tasks/jumper/dance/export_media.py` are drivers of it, and ffmpeg is
    still allowed to them for what comes *after* the MP4 (a GIF, a soundtrack).

    The control group is the recorder itself, which must contain the call: an
    empty offender list means nothing if the pattern stopped matching anything.
    """
    pattern = re.compile(r"\bwrite_frames\(")
    recorder = REPO / "rl/mjrl/viewer/video.py"
    assert pattern.search(recorder.read_text(encoding="utf-8")), (
        "the recorder no longer calls imageio_ffmpeg.write_frames; "
        "move this test's pattern with it"
    )
    offenders = [
        str(p.relative_to(REPO))
        for p in first_party_py()
        if p != recorder and "tests" not in p.parts
        and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "these write MP4 frames themselves instead of running `play.py --video`: "
        + ", ".join(offenders)
    )
