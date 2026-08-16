from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HYGIENE_SCAN = ROOT / "scripts" / "hygiene_scan.sh"


def _run_hygiene(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HYGIENE_SCAN), str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE)
    (path / ".gitignore").write_text(".venv/\nbuild/\ndist/\n*.egg-info/\n", encoding="utf-8")
    (path / "README.md").write_text("# temporary repo\n", encoding="utf-8")


def test_hygiene_scan_ignores_local_virtualenv(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    venv_data = tmp_path / ".venv" / "lib" / "python" / "site-packages" / "pkg"
    venv_data.mkdir(parents=True)
    (venv_data / "weights.safetensors").write_text("not public\n", encoding="utf-8")
    (venv_data / "token.txt").write_text("local_credential=not-real\n", encoding="utf-8")

    result = _run_hygiene(tmp_path)

    assert result.returncode == 0, result.stdout


def test_hygiene_scan_flags_public_model_residue(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "weights.safetensors").write_text("not public\n", encoding="utf-8")

    result = _run_hygiene(tmp_path)

    assert result.returncode == 1
    assert "Model artifacts do not belong in Git" in result.stdout
    assert "Private workspace residue found" in result.stdout
