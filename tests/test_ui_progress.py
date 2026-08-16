from __future__ import annotations

from mtplx.ui.progress import ModelLoadProgress


def test_model_load_progress_prints_plain_status_and_stops_heartbeat(monkeypatch, capsys):
    calls: list[dict] = []

    class FakeProc:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            self.killed = True

    proc = FakeProc()

    def fake_popen(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return proc

    monkeypatch.setattr("mtplx.ui.progress.subprocess.Popen", fake_popen)

    with ModelLoadProgress("Loading model", interval_s=10.0) as progress:
        progress.set_subtitle("profile performance-cold")
        progress.set_subtitle("ready")

    captured = capsys.readouterr().out
    assert "[mtplx] Model load in progress (this may take a minute)." in captured
    assert "[mtplx] Loading model: profile performance-cold" in captured
    assert "Loading model: ready" not in captured
    assert calls
    assert calls[0]["cmd"][3] == "10.0"
    assert proc.terminated is True
    assert proc.killed is False
