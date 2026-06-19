"""Symmetry enumeration and Burnside counting for binary colorings.

Usage::

    from archimono.assignment import enumeration

    enum = enumeration.get()        # OrderlyEnumerator (default backend)
    enum = enumeration.get("icet")  # IcetEnumerator
"""

from archimono.assignment.enumeration._helpers import (
    _get_symmetry_permutations,
    _tiling_to_atoms,
    assignment_to_atoms,
)
from archimono.assignment.enumeration.icet import IcetEnumerator
from archimono.assignment.enumeration.orderly import OrderlyEnumerator

#: Public alias for building an ASE ``Atoms`` template from a tiling supercell.
#: Stable entry point so callers (e.g. :mod:`archimono.export`) need not reach
#: into the private ``_helpers`` module.
tiling_to_atoms = _tiling_to_atoms

#: Default enumeration backend. Orderly generation is the icet-free
#: pure-Python enumerator and is the canonical default for the package.
DEFAULT_BACKEND = "orderly"

_BACKENDS: dict[str, type[OrderlyEnumerator] | type[IcetEnumerator]] = {
    "orderly": OrderlyEnumerator,
    "icet": IcetEnumerator,
}


def get(backend: str = DEFAULT_BACKEND) -> OrderlyEnumerator | IcetEnumerator:
    """Return an enumeration backend instance.

    The default backend is orderly generation
    (:class:`~archimono.assignment.enumeration.orderly.OrderlyEnumerator`),
    the icet-free pure-Python enumerator. Pass ``"icet"`` to select the
    icet-based :class:`~archimono.assignment.enumeration.icet.IcetEnumerator`
    instead. Mirrors :func:`archimono.tilings.registry.get`.

    Args:
        backend: Backend name, one of ``"orderly"`` or ``"icet"``.
            Defaults to ``"orderly"`` (:data:`DEFAULT_BACKEND`).

    Returns:
        A freshly constructed enumerator instance for *backend*.

    Raises:
        KeyError: If *backend* is not a recognised backend name.
    """
    if backend not in _BACKENDS:
        available = sorted(_BACKENDS)
        raise KeyError(
            f"Unknown enumeration backend '{backend}'. Available: {available}"
        )
    return _BACKENDS[backend]()


__all__ = [
    "DEFAULT_BACKEND",
    "IcetEnumerator",
    "OrderlyEnumerator",
    "assignment_to_atoms",
    "get",
    "tiling_to_atoms",
    "_get_symmetry_permutations",
]
