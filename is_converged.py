"""Decide whether a JDFTx calculation is converged, from its out file.

A calculation counts as converged only if every check below passes, in this order:

1. charge                |net charge| < MAX_ABS_CHARGE electrons
2. energy                final energy < 0
3. ionic_converged       JDFTx marked the geometry optimization converged
4. electronic_converged  JDFTx marked the final electronic minimization converged
5. electronic_reason     the electronic convergence reason does not mention roundoff
6. ionic_reason          the ionic convergence reason does not mention roundoff
7. fmax                  largest force on a free atom in the last step with forces < FMAX_TOL eV/A

Checks 1-6 are the tests (and order) of NO3_ads_gen/gen/from_aziz/fix_offrailed_calcs.py: checks 1-2 catch runs that
have gone off the rails, and the convergence flags of such a run mean nothing. fmax follows the project's residual
force (NO3_ads_gen/anl/d3_c6_fix_impact/analyze.py): the largest force norm over the atoms the optimizer may move. A
value the out file does not give (no forces, no electron counts, no energy) fails its check.

JDFTx sets the ionic flag only when its own ionic minimizer converges. Runs driven by ASE (one JDFTx single point per
step) and single-point runs therefore never pass check 3.

Usage:
    python is_converged.py OUT [OUT ...]    # one line per file; exit status 0 only if every file is converged

    from is_converged import is_converged, get_checks
    is_converged("path/to/out")             # True or False; also takes a parsed JDFTXOutfile

On import, this file compares itself with the latest version on GitHub (LATEST_URL). An out-of-date copy replaces
itself with the GitHub version, which runs from the next start. Only a copy that once matched GitHub is replaced, so
a copy with local edits just gets a note on stderr. Once a copy is found identical, it is not checked again for
CHECK_INTERVAL (24 h): the stamp is an empty file in STAMP_DIR named after the copy's SHA-256, so editing or updating
the copy starts checking again. Without internet the check does nothing. Environment variables (none needed):
IS_CONVERGED_AUTO_UPDATE=0 only prints the note; IS_CONVERGED_SKIP_UPDATE_CHECK=1 skips the check (e.g. to avoid the
network timeout on compute nodes without internet).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from pymatgen.io.jdftx.outputs import JDFTXOutfile

MAX_ABS_CHARGE = 10.0  # electrons
FMAX_TOL = 0.1  # eV/A
LATEST_URL = "https://raw.githubusercontent.com/benrich37/is_converged/main/is_converged.py"
HISTORY_URL = "https://github.com/benrich37/is_converged/commits/main/is_converged.py"
SKIP_VAR = "IS_CONVERGED_SKIP_UPDATE_CHECK"
AUTO_VAR = "IS_CONVERGED_AUTO_UPDATE"
CHECK_INTERVAL = 24 * 3600  # s
STAMP_DIR = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "is_converged"


def replace_file(path: Path, new: bytes) -> None:
    """Atomically replace path with new, refusing anything that is not a complete, compilable copy of this script."""
    compile(new, LATEST_URL, "exec")  # an error page, or a broken push that no copy could recover from
    if not new.rstrip().endswith(b"sys.exit(main())"):  # an empty or cut-off download can still compile
        raise ValueError("the download is not a complete is_converged.py")
    if not os.access(path, os.W_OK):  # os.replace would ignore a read-only file
        raise PermissionError(f"{path} is read-only")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def check_for_update(timeout: float = 2.0, auto_update: bool = True) -> None:
    """Update this file from LATEST_URL (or print a note) if it differs; silent if GitHub cannot be reached.

    Skipped within CHECK_INTERVAL of finding this copy identical.

    Never raises, and prints rather than warns, so warning filters can neither hide the note nor turn it into an
    error. Runs once per process tree: child processes inherit the skip flag set here. urlopen's timeout applies to
    each address tried, so a network that drops packets can take several timeouts. With auto_update, a differing copy
    is replaced by the GitHub version if a stamp shows this exact copy once matched GitHub (so its content is in the
    repo's history); this process keeps running the old code.
    """
    if os.environ.get(SKIP_VAR):
        return
    os.environ[SKIP_VAR] = "1"
    try:
        path = Path(__file__).resolve()
        local = path.read_bytes()
        stamp = STAMP_DIR / hashlib.sha256(local).hexdigest()
        if stamp.exists() and time.time() - stamp.stat().st_mtime < CHECK_INTERVAL:
            return
        with urllib.request.urlopen(LATEST_URL, timeout=timeout) as response:
            latest = response.read()
    except Exception:  # no network, no __file__ (code pasted into a notebook), unreadable file: stay silent
        return
    if local.splitlines() != latest.splitlines():
        note = (f"is_converged: {path} differs from the version on GitHub (out of date, or edited locally)\n"
                f"  changes: {HISTORY_URL}\n  update:  curl -o '{path}' {LATEST_URL}")
        if not auto_update:
            print(note, file=sys.stderr)
            return
        try:
            if not stamp.exists():
                raise ValueError("this copy never matched GitHub (edited locally, or newer than GitHub)")
            replace_file(path, latest)
        except Exception as e:
            print(f"{note}\n  not updated automatically ({type(e).__name__}: {e})", file=sys.stderr)
            return
        print(f"is_converged: updated {path} to the version on GitHub; it runs from the next start\n"
              f"  changes: {HISTORY_URL}", file=sys.stderr)
        stamp = STAMP_DIR / hashlib.sha256(latest).hexdigest()
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
    except OSError:  # unwritable cache dir: check again next time
        pass


check_for_update(auto_update=os.environ.get(AUTO_VAR) != "0")


def get_charge(outfile: JDFTXOutfile) -> float | None:
    """Net charge (neutral electron count minus electron count, so positive is cationic), None if not printed."""
    if None in (outfile.total_electrons_uncharged, outfile.total_electrons):
        return None
    return outfile.total_electrons_uncharged - outfile.total_electrons


def get_fmax(outfile: JDFTXOutfile) -> float | None:
    """Largest force norm (eV/A) on a free atom (moveScale != 0) in the last step that printed forces, None if none.

    The last step of a run that was cut off may have no forces yet; pymatgen then gives the step before it.
    """
    if outfile.forces is None:
        return None
    norms = np.linalg.norm(outfile.forces, axis=1)
    if outfile.jstrucs.selective_dynamics is not None:
        norms = norms[np.any(np.asarray(outfile.jstrucs.selective_dynamics) != 0, axis=1)]
    return float(norms.max(initial=0.0))


def mentions_roundoff(reason: str | None) -> bool:
    return isinstance(reason, str) and "roundoff" in reason.lower()


def get_checks(outfile: JDFTXOutfile) -> dict[str, tuple[bool, object]]:
    """Every check, in order, as name -> (passed, value tested)."""
    jstrucs = outfile.jstrucs
    ion_converged = jstrucs.geom_converged if jstrucs else False
    ion_reason = jstrucs.geom_converged_reason if jstrucs else None
    elec_converged = jstrucs.elec_converged if jstrucs else False
    elec_reason = jstrucs.elec_converged_reason if jstrucs else None
    charge = get_charge(outfile)
    energy = outfile.e
    fmax = get_fmax(outfile)
    return {
        "charge": (charge is not None and abs(charge) < MAX_ABS_CHARGE, charge),
        "energy": (energy is not None and energy < 0.0, energy),
        "ionic_converged": (bool(ion_converged), ion_converged),
        "electronic_converged": (bool(elec_converged), elec_converged),
        "electronic_reason": (not mentions_roundoff(elec_reason), elec_reason),
        "ionic_reason": (not mentions_roundoff(ion_reason), ion_reason),
        "fmax": (fmax is not None and fmax < FMAX_TOL, fmax),
    }


def is_converged(outfile: JDFTXOutfile | str | Path) -> bool:
    if not isinstance(outfile, JDFTXOutfile):
        outfile = JDFTXOutfile.from_file(outfile)
    return all(passed for passed, _ in get_checks(outfile).values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outfiles", nargs="+", type=Path, help="JDFTx out file(s)")
    args = parser.parse_args()
    all_converged = True
    for path in args.outfiles:
        try:
            checks = get_checks(JDFTXOutfile.from_file(path))
        except Exception as e:  # an unreadable out file is not a converged calculation
            print(f"{path}: NOT CONVERGED (could not parse: {type(e).__name__}: {e})")
            all_converged = False
            continue
        failed = [f"{name}={value!r}" for name, (passed, value) in checks.items() if not passed]
        print(f"{path}: NOT CONVERGED ({', '.join(failed)})" if failed else f"{path}: CONVERGED")
        all_converged = all_converged and not failed
    return 0 if all_converged else 1


if __name__ == "__main__":
    sys.exit(main())
