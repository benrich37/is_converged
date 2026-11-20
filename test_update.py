"""Test of is_converged.py's auto-update against a local "GitHub" (file:// URL): no network, no real ~/.cache.

Run: python test_update.py  (or pytest)
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SRC = (Path(__file__).parent / "is_converged.py").read_text()
NEW = SRC.replace("FMAX_TOL = 0.1", "FMAX_TOL = 0.2")
URL_LINE = next(line for line in SRC.splitlines() if line.startswith("LATEST_URL = "))


def update(remote: str | None, stamped: bool = True, auto: str | None = None) -> tuple[str, str]:
    """Import a copy of SRC that sees `remote` as the GitHub version (None: unreachable); return its text and stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if remote is not None:
            (tmp / "remote.py").write_text(remote)
        copy = tmp / "is_converged.py"
        copy.write_text(SRC.replace(URL_LINE, f'LATEST_URL = "file://{tmp / "remote.py"}"'))
        if stamped:  # this copy matched GitHub over a day ago
            stamp = tmp / "cache" / "is_converged" / hashlib.sha256(copy.read_bytes()).hexdigest()
            stamp.parent.mkdir(parents=True)
            stamp.touch()
            os.utime(stamp, (time.time() - 25 * 3600,) * 2)
        env = {k: v for k, v in os.environ.items() if not k.startswith("IS_CONVERGED_")}
        env["XDG_CACHE_HOME"] = str(tmp / "cache")
        if auto is not None:
            env["IS_CONVERGED_AUTO_UPDATE"] = auto
        code = "import warnings; warnings.simplefilter('ignore'); import is_converged"  # pymatgen's own warnings
        run = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, check=True, capture_output=True, text=True)
        return copy.read_text(), run.stderr


def test_auto_update():
    assert update(NEW)[0] == NEW  # on by default: a copy that once matched GitHub becomes the GitHub version
    text, stderr = update(None)
    assert "FMAX_TOL = 0.1" in text and stderr == "", stderr  # GitHub unreachable: unchanged and silent
    kept = {
        "auto-update off": update(NEW, auto="0")[0],
        "never matched GitHub (local edit)": update(NEW, stamped=False)[0],
        "error page": update("<html>404</html>")[0],
        "download cut at a line end": update(NEW.rsplit("\nif __name__", 1)[0])[0],
        "download cut mid-statement": update(NEW[: len(NEW) // 2])[0],
    }
    for case, text in kept.items():
        assert "FMAX_TOL = 0.1" in text and "FMAX_TOL = 0.2" not in text, case


if __name__ == "__main__":
    test_auto_update()
    print("ok")
