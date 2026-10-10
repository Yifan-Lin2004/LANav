import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_config_contract():
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_paper_config.py")],
        cwd=ROOT,
        check=True,
    )


def test_release_contains_no_other_policy_family():
    forbidden = ("da_wsla", "transformer", "detection", "siglip_det")
    scoped_files = list((ROOT / "ovon" / "models").glob("*.py")) + list(
        (ROOT / "config" / "experiments").glob("*.yaml")
    )
    relative_paths = [str(path.relative_to(ROOT)).lower() for path in scoped_files]
    offenders = [path for path in relative_paths if any(word in path for word in forbidden)]
    assert offenders == []
