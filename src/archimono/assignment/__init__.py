"""Species assignment: MAX-CUT solvers, symmetry enumeration, and Burnside counting."""

import sys

from archimono.assignment.enumeration import (
    burnside,
    canonical,
    icet,
    orderly,
)
from archimono.assignment.solvers import (
    _bruteforce,
    _heuristic_base,
    _scoring,
    _validation,
    annealing,
    base,
    bipartite,
    bruteforce,
    frontier_dp,
    greedy,
    hadlock,
)

AssignmentResult = base.AssignmentResult
AssignmentSolver = base.AssignmentSolver
AnnealingSolver = annealing.AnnealingSolver
BipartiteAssigner = bipartite.BipartiteAssigner
BruteforceSolver = bruteforce.BruteforceSolver
BurnsideCounter = burnside.BurnsideCounter
FrontierExactSolver = frontier_dp.FrontierExactSolver
FrontierExactSolverCK = frontier_dp.FrontierExactSolverCK
FrontierExactSolverGreedy = frontier_dp.FrontierExactSolverGreedy
FrontierExactSolverNatural = frontier_dp.FrontierExactSolverNatural
compute_frontier_width = frontier_dp.compute_frontier_width
compute_max_cut_value = frontier_dp.compute_max_cut_value
estimate_peak_memory_bytes = frontier_dp.estimate_peak_memory_bytes
select_heuristic = frontier_dp.select_heuristic
GreedySolver = greedy.GreedySolver
HadlockSolver = hadlock.HadlockSolver
IcetEnumerator = icet.IcetEnumerator
OrderlyEnumerator = orderly.OrderlyEnumerator
canonical_form = canonical.canonical_form
filter_active = canonical.filter_active
prepare_active = canonical.prepare_active

# Backwards-compatible module paths: code like
# ``from archimono.assignment.annealing import AnnealingSolver``
# or ``from archimono.assignment.burnside import BurnsideCounter``
# continues to work via sys.modules aliasing.
_compat = {
    f"{__name__}.{alias}": mod
    for alias, mod in {
        "annealing": annealing,
        "base": base,
        "bipartite": bipartite,
        "bruteforce": bruteforce,
        "frontier_dp": frontier_dp,
        "greedy": greedy,
        "hadlock": hadlock,
        "_bruteforce": _bruteforce,
        "_heuristic_base": _heuristic_base,
        "_scoring": _scoring,
        "_validation": _validation,
        "burnside": burnside,
        "canonical": canonical,
        "orderly": orderly,
    }.items()
}
sys.modules.update(_compat)

__all__ = [
    "AssignmentResult",
    "AssignmentSolver",
    "AnnealingSolver",
    "BipartiteAssigner",
    "BruteforceSolver",
    "BurnsideCounter",
    "FrontierExactSolver",
    "FrontierExactSolverCK",
    "FrontierExactSolverGreedy",
    "FrontierExactSolverNatural",
    "compute_frontier_width",
    "compute_max_cut_value",
    "estimate_peak_memory_bytes",
    "select_heuristic",
    "GreedySolver",
    "HadlockSolver",
    "IcetEnumerator",
    "OrderlyEnumerator",
    "canonical_form",
    "filter_active",
    "prepare_active",
]
