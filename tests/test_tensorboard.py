"""TensorBoard started alongside training.

Training already wrote the event files; what this adds is starting the server on
the right directory without anyone typing a path. Both of those are things that
**fail without raising**:

- Pointed one level too high or too low, the page loads, shows a run, and it is
  the wrong one -- or none at all, which reads as "training is not logging".
- A server that died on startup still had its URL printed, so the browser shows
  a connection error that looks like a network problem.

So the assertions here are about the directory being exactly right and about
failures being reported rather than announced as success.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
from pathlib import Path

import pytest

from mjrl.viewer.tensorboard import (
    TensorBoard,
    _can_open_browser,
    _free_port,
    _port_is_open,
    resolve_logdir,
    sigterm_runs_finally,
)
from tasks.paths import LOGS_DIRNAME, log_root_for

REPO = Path(__file__).resolve().parents[1]


def _run_dir(task: str = "jumper.flat", stamp: str = "2026-09-04_12-00-00") -> Path:
    """A run directory built the way scripts/train.py builds it."""
    return Path(log_root_for(REPO / "assets" / "jumper" / "jumper.xml")) / task / stamp


# ── The directory it serves ─────────────────────────────────────────────


def test_scopes_match_the_real_log_layout() -> None:
    """The three scopes are the three levels of `logs/<model>/<task>/<stamp>`.

    Tied to a path built by `tasks.paths.log_root_for` rather than a literal, so
    a change to the layout fails here -- `resolve_logdir` walks up by position
    and would otherwise go on serving a level that has quietly become the wrong
    one.
    """
    d = _run_dir()
    assert d.parts == (LOGS_DIRNAME, "jumper", "jumper.flat", "2026-09-04_12-00-00"), (
        "the layout this module walks up through has changed"
    )
    assert resolve_logdir(d, "run") == d
    assert resolve_logdir(d, "task") == Path(LOGS_DIRNAME) / "jumper" / "jumper.flat"
    assert resolve_logdir(d, "all") == Path(LOGS_DIRNAME)


def test_default_scope_holds_the_other_runs_of_this_task() -> None:
    """The default scope is what makes the previous run visible.

    TensorBoard names each subdirectory as a run, so serving the task directory
    labels the series by timestamp and puts the last run alongside this one. The
    `run` scope shows a single unnamed series and nothing to compare against,
    which is why it is not the default.
    """
    today, yesterday = _run_dir(stamp="2026-09-04_12-00-00"), _run_dir(stamp="2026-09-03_09-30-00")
    served = resolve_logdir(today, "task")
    assert served == resolve_logdir(yesterday, "task")
    assert today.parent == served and yesterday.parent == served
    # And the control: the run scope keeps them apart, which is the point of
    # having the option at all.
    assert resolve_logdir(today, "run") != resolve_logdir(yesterday, "run")


def test_unknown_scope_raises() -> None:
    """An unrecognised scope must not fall back to some default level.

    Silently serving `logs/` for a typo would look like it worked -- the page
    loads and shows runs, just not the ones asked for.
    """
    with pytest.raises(ValueError, match="unknown scope"):
        resolve_logdir(_run_dir(), "runs")


# ── Turning it off ──────────────────────────────────────────────────────


def _parse(argv: list[str], env: str | None, monkeypatch) -> argparse.Namespace:
    """Parse the TensorBoard switches exactly as `scripts/train.py` does.

    `scripts/` is not a package and nothing may put it on `sys.path` (see
    `test_no_sys_path_mutation` in test_layout.py), so `_cli` is loaded from its
    file. Going through the real parser is the point: the default is computed
    from the environment *inside* `add_tensorboard_args`, and that computation --
    not the code that reads `args.tensorboard` -- is where this has gone wrong.
    """
    if env is None:
        monkeypatch.delenv("MJRL_TENSORBOARD", raising=False)
    else:
        monkeypatch.setenv("MJRL_TENSORBOARD", env)
    spec = importlib.util.spec_from_file_location("_cli_under_test", REPO / "scripts" / "_cli.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    parser = cli.build_parser("train.py", "")
    cli.add_tensorboard_args(parser)
    return parser.parse_args(argv)


def test_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing said either way it starts -- that is the feature."""
    assert _parse([], None, monkeypatch).tensorboard is True


def test_the_switch_turns_it_off_for_one_run(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse(["--no-tensorboard"], None, monkeypatch).tensorboard is False


@pytest.mark.parametrize("value", ["off", "OFF", "false", "no", "0"])
def test_the_env_key_turns_it_off_for_good(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every spelling of "off" must work, not just the lowercase one.

    `MJRL_TENSORBOARD=OFF` used to leave the server running -- the default was
    `!= "off"`. Nothing in the output mentions the setting, so the only available
    conclusion was that `.env` is not read at all.
    """
    assert _parse([], value, monkeypatch).tensorboard is False


def test_the_command_line_wins_in_both_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--tensorboard` must beat `MJRL_TENSORBOARD=off`.

    The precedence the whole configuration rests on is "command line > .env". With
    only a negative switch that held in one direction: a machine whose `.env` said
    `off` had no way to see curves for a single run without editing the file.
    """
    assert _parse(["--tensorboard"], "off", monkeypatch).tensorboard is True
    assert _parse(["--no-tensorboard"], "on", monkeypatch).tensorboard is False


def test_a_value_that_is_not_a_switch_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not silently mean "on"; it exits, naming the value."""
    with pytest.raises(SystemExit):
        _parse([], "of", monkeypatch)


# ── Choosing a port ─────────────────────────────────────────────────────


def test_free_port_ignores_a_lingering_time_wait() -> None:
    """A port in TIME_WAIT is free as far as the server is concerned.

    Without `SO_REUSEADDR` the probe answers a different question than the one
    being asked. Every previous run leaves TIME_WAIT entries on its port (the
    browser tab held a connection when the server was terminated), so the probe
    would skip a perfectly usable 6006 and the URL would climb 6006 -> 6007 ->
    6008 run after run, which makes a bookmark worthless.
    """
    srv = socket.socket()
    # The server this stands in for sets SO_REUSEADDR on its listening socket
    # (socketserver's `allow_reuse_address`, which TensorBoard's werkzeug
    # inherits), and the connections it accepts carry the option into TIME_WAIT.
    # That matters on Linux, where a bind with SO_REUSEADDR over a TIME_WAIT
    # entry succeeds only if that entry has the option too; BSD and macOS ask
    # only the new socket. Without this line the test passed on macOS and
    # failed on every Linux machine (reported 49606 for 49605 in a container on
    # 2026-10-08) while probing a port no real server would have left behind.
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    cli = socket.socket()
    cli.connect(("127.0.0.1", port))
    conn, _ = srv.accept()
    # Closing the *accepted* side first is what puts the listening port into
    # TIME_WAIT -- the same order as terminating a server with a tab open on it.
    conn.close()
    srv.close()
    cli.close()

    assert _free_port(port) == port, "skipped a port that was only in TIME_WAIT"


def test_free_port_skips_a_port_someone_is_serving() -> None:
    """The control group: SO_REUSEADDR must not make the probe blind.

    It bypasses TIME_WAIT only; two sockets still cannot listen on one port
    (that would be SO_REUSEPORT). Were this to pass with the same port, the
    second run would hand out a URL to the first run's server.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert _free_port(port) != port, "handed out a port already being served"
    finally:
        srv.close()


# ── Reporting failure instead of announcing success ─────────────────────


def test_a_server_that_died_is_reported(tmp_path: Path) -> None:
    """A process that exited must raise, not be waited on until the timeout.

    The caller prints the URL as soon as `start()` returns, so returning happily
    here is what produces "the page will not load" -- which reads as a network
    problem rather than a process that never came up. The exit code and the tail
    of its own log go into the message because that is the only place the reason
    exists.
    """
    log = tmp_path / "tensorboard.log"
    log.write_text("Traceback ...\nOSError: [Errno 48] Address already in use\n")

    class _Exited:
        returncode = 1

        def poll(self):
            return 1

    tb = TensorBoard(tmp_path, port=1, log_file=log)
    tb._proc = _Exited()
    with pytest.raises(RuntimeError, match="exited with code 1"):
        tb._wait_until_ready()


def test_an_open_port_is_not_proof_the_server_is_ours(tmp_path: Path) -> None:
    """Readiness is this server's announcement, not merely an open port.

    An open port answers "is anything here", which is a different question. Take
    it as readiness and a stranger's server on that port masks ours dying: the
    URL gets printed, the page loads, and it shows somebody else's runs.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        log = tmp_path / "tensorboard.log"
        log.write_text("")
        tb = TensorBoard(tmp_path, port, log)
        assert _port_is_open(port), "the premise: something is answering on it"
        assert not tb._announced_its_url(), "an open port must not count as ready"

        # The control: once the server says it bound *this* address, it is ready.
        log.write_text(f"TensorBoard 2.21.0 at http://127.0.0.1:{port}/ (Press CTRL+C to quit)\n")
        assert tb._announced_its_url()

        # And an announcement for some other port is not this one's.
        log.write_text(f"TensorBoard 2.21.0 at http://127.0.0.1:{port + 1}/\n")
        assert not tb._announced_its_url()
    finally:
        srv.close()


def test_start_fails_when_the_port_is_taken(tmp_path: Path) -> None:
    """End to end: a server that cannot bind must raise, not report success.

    This launches the real thing, because the bug it guards against only exists
    in the seam between the two -- with the readiness check satisfied by the
    *occupying* server, the process exited with "Address already in use" while
    `start()` returned happily.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    tb = TensorBoard(tmp_path, port, tmp_path / "tensorboard.log")
    try:
        with pytest.raises(RuntimeError, match="exited with code"):
            tb.start()
        # The reason has to survive into the message: it is the only place it exists.
        assert "already in use" in tb._tail(lines=5)
    finally:
        tb.stop()
        srv.close()


def test_stop_is_safe_before_and_after_start(tmp_path: Path) -> None:
    """`stop()` runs on the way out of a failed run too, so it must never raise."""
    tb = TensorBoard(tmp_path, port=1, log_file=tmp_path / "tb.log")
    tb.stop()  # never started
    tb.stop()  # and again


def test_failure_never_takes_the_run_down() -> None:
    """A server that will not start costs a convenience, not an experiment.

    The same rule as the live viewer: a run that died because port 6006 was
    taken would be a far worse trade than not seeing curves.
    """
    cli = (REPO / "scripts" / "_cli.py").read_text(encoding="utf-8")
    body = cli.split("def maybe_tensorboard(", 1)[1]
    assert "except Exception" in body, "a failure to start must not propagate"
    assert "yield None" in body, "and the run must go on without it"


# ── Not leaving the server behind ───────────────────────────────────────


def test_sigterm_runs_the_finally_blocks() -> None:
    """SIGTERM has to unwind rather than kill outright.

    The server is a **child process**, so anything that stops training without
    running its `finally` blocks leaves it behind holding the port and serving
    the *previous* run -- a bookmark that opens on stale curves looking like a
    training that stopped improving. `kill <pid>` is exactly that, and it is how
    background runs are stopped here.
    """
    import os
    import signal

    stopped = []
    with pytest.raises(SystemExit):
        with sigterm_runs_finally():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            finally:
                stopped.append("cleanup ran")
    assert stopped == ["cleanup ran"]


def test_sigterm_disposition_is_restored() -> None:
    """Outside the block, SIGTERM must behave exactly as it did before.

    Training's response to signals is not this feature's to change permanently;
    the guard exists only while there is a child process to clean up.
    """
    import signal

    before = signal.getsignal(signal.SIGTERM)
    with sigterm_runs_finally():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


# ── Opening a browser ───────────────────────────────────────────────────


def test_no_browser_over_ssh(monkeypatch) -> None:
    """Over ssh the page would open on the remote console, where nobody is.

    This check has to come before the platform check: on a remote macOS box
    `sys.platform` alone says yes.
    """
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    assert not _can_open_browser()
    monkeypatch.delenv("SSH_CONNECTION")
    assert _can_open_browser(), "the control: locally on macOS it should open"


def test_no_browser_without_a_display(monkeypatch) -> None:
    """A headless Linux training box has no browser to open."""
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not _can_open_browser()
    monkeypatch.setenv("DISPLAY", ":0")
    assert _can_open_browser()


# ── Wiring ──────────────────────────────────────────────────────────────


def test_it_starts_before_the_environment_is_built() -> None:
    """The page should be up during the slow part, not after it.

    Building the scene and compiling one model per environment takes tens of
    seconds, and that is exactly the window in which someone wants the tab
    already open. Start it afterwards and the feature is worth much less.
    """
    src = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    assert src.index("maybe_tensorboard(") < src.index("ManagerBasedRlEnv(cfg=")


def test_config_is_actually_wired() -> None:
    """`MJRL_TENSORBOARD` / `MJRL_TB_PORT` have to reach the code.

    `MJRL_CPU_THREADS` was once parsed, computed, printed and never passed on.
    **A config that lies is worse than no config**: it makes people believe they
    have turned something off.
    """
    cli = (REPO / "scripts" / "_cli.py").read_text(encoding="utf-8")
    # Read as a switch, not as a string: `_env(...) != "off"` left every spelling
    # but the lowercase one meaning "on". (Spelt out because `_bool_env(` happens
    # to contain `_env(`, so the looser check passed either way.)
    assert '_bool_env("MJRL_TENSORBOARD"' in cli
    assert '_int_env("MJRL_TB_PORT"' in cli
    env = (REPO / ".env").read_text(encoding="utf-8")
    assert "MJRL_TENSORBOARD=" in env and "MJRL_TB_PORT=" in env

    train = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "args.tb_scope" not in train, "train.py should not reach past the helper"
    assert "maybe_tensorboard(log_dir, args)" in train


def test_it_binds_to_localhost_only() -> None:
    """Training metrics must not be served to the whole network by default.

    Over ssh the answer is to forward the port, not to bind 0.0.0.0.
    """
    src = (REPO / "rl" / "mjrl" / "viewer" / "tensorboard.py").read_text(encoding="utf-8")
    start = src.split("def start(", 1)[1].split("\n    def ", 1)[0]
    assert '"--host", "127.0.0.1"' in start
    assert "--bind_all" not in start


def test_the_server_comes_from_this_interpreter() -> None:
    """`sys.executable -m` rather than the `tensorboard` console script.

    The script is on PATH only when the virtualenv is activated, and training is
    routinely started as `.venv/bin/python scripts/train.py` without activating
    anything. Going through the interpreter also guarantees the server is the
    one from the same environment as the writer.
    """
    src = (REPO / "rl" / "mjrl" / "viewer" / "tensorboard.py").read_text(encoding="utf-8")
    start = src.split("def start(", 1)[1].split("\n    def ", 1)[0]
    assert 'sys.executable, "-m", "tensorboard.main"' in start
