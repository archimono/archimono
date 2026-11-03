"""Species assignment: MAX-CUT solvers, symmetry enumeration, and Burnside counting."""

from archimono.assignment.enumeration import (
    burnside,
    canonical,
    icet,
    orderly,
)
from archimono.assignment.solvers import (
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
compute_dp_cost = frontier_dp.compute_dp_cost
compute_frontier_sizes = frontier_dp.compute_frontier_sizes
compute_frontier_width = frontier_dp.compute_frontier_width
compute_max_cut_value = frontier_dp.compute_max_cut_value
estimate_peak_memory_bytes = frontier_dp.estimate_peak_memory_bytes
estimate_reconstruct_memory_bytes = frontier_dp.estimate_reconstruct_memory_bytes
select_heuristic = frontier_dp.select_heuristic
GreedySolver = greedy.GreedySolver
HadlockSolver = hadlock.HadlockSolver
IcetEnumerator = icet.IcetEnumerator
OrderlyEnumerator = orderly.OrderlyEnumerator
canonical_form = canonical.canonical_form
filter_active = canonical.filter_active
prepare_active = canonical.prepare_active

#: Canonical mapping from CLI solver short-name to solver class. Single source
#: of truth for solver-name dispatch; downstream call sites derive their own
#: views (instance factory, class-name string) from this.
SOLVERS_BY_NAME: dict[str, type[AssignmentSolver]] = {
    "bruteforce": BruteforceSolver,
    "frontier-dp": FrontierExactSolver,
    "annealing": AnnealingSolver,
    "greedy": GreedySolver,
    "hadlock": HadlockSolver,
}

__all__ = [
    "AssignmentResult",
    "AssignmentSolver",
    "AnnealingSolver",
    "BipartiteAssigner",
    "BruteforceSolver",
    "BurnsideCounter",
    "FrontierExactSolver",
    "compute_dp_cost",
    "compute_frontier_sizes",
    "compute_frontier_width",
    "compute_max_cut_value",
    "estimate_peak_memory_bytes",
    "estimate_reconstruct_memory_bytes",
    "select_heuristic",
    "GreedySolver",
    "HadlockSolver",
    "IcetEnumerator",
    "OrderlyEnumerator",
    "canonical_form",
    "filter_active",
    "prepare_active",
    "SOLVERS_BY_NAME",
]
