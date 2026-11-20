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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pymatgen.io.jdftx.outputs import JDFTXOutfile

MAX_ABS_CHARGE = 10.0  # electrons
FMAX_TOL = 0.1  # eV/A


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
