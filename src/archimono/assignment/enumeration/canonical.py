"""Canonical-form utilities for binary colorings under permutation groups.

Provides reusable primitives for orbit canonicalization at two levels:

1. **Full canonicalization** — :func:`canonical_form` computes the
   lexicographically smallest group-image of a complete assignment.
2. **Incremental canonicalization** — :func:`filter_active` performs a
   single-step prefix comparison for backtracking algorithms that build
   assignments site-by-site.

Both use the **pull-back** convention: for a permutation *g* acting on an
assignment *a*, the image is ``transformed[i] = a[g[i]]``.  The identity
permutation is never informative (its image always equals the input), so
:func:`prepare_active` strips it from the group before incremental use.

References:
    McKay, B. D. (1998).
        Isomorph-free exhaustive generation.
        *Journal of Algorithms*, 26(2), 306–324.
        https://doi.org/10.1006/jagm.1997.0898
        §3: canonical deletion and prefix-based pruning.
"""

from __future__ import annotations


def canonical_form(
    assignment: tuple[int, ...],
    group: list[tuple[int, ...]],
) -> tuple[int, ...]:
    """Return the lexicographically smallest orbit representative.

    For each symmetry operation *g*, the transformed assignment is
    ``transformed[i] = assignment[g[i]]`` (pull-back).  The canonical
    form is the minimum over all such transforms.

    Args:
        assignment: A binary tuple of length *n*
            (0 = ``species[0]``, 1 = ``species[1]``).
        group: Symmetry group as a list of permutation tuples.

    Returns:
        The canonical (lex-minimum) representative of the orbit.
    """
    canon = assignment
    for perm in group:
        transformed = tuple(
            assignment[perm[i]] for i in range(len(assignment))
        )
        if transformed < canon:
            canon = transformed
    return canon


def prepare_active(
    group: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Strip the identity from *group* to produce an initial active set.

    The identity permutation maps every assignment to itself, so it never
    prunes a branch and never becomes dead.  Excluding it avoids a no-op
    comparison at every backtracking step.

    Args:
        group: Symmetry group as a list of permutation tuples.  All
            tuples must have the same length.  An empty list returns
            an empty active set.

    Returns:
        All non-identity elements of *group*.
    """
    if not group:
        return []
    n = len(group[0])
    identity = tuple(range(n))
    return [g for g in group if g != identity]


def filter_active(
    active: list[tuple[int, ...]],
    assignment: list[int],
    pos: int,
    value: int,
) -> tuple[list[tuple[int, ...]], bool]:
    """One-step incremental canonical check for backtracking algorithms.

    Given the current active set of non-identity group elements,
    determine which elements survive after assigning *value* at position
    *pos*, and whether the branch remains potentially canonical.

    For each active element *g*, positions 0 through *pos* are scanned
    left-to-right (matching lexicographic order).  At each position *j*:

    - If ``g[j] <= pos`` (resolvable): compare the image value
      ``assignment[g[j]]`` (or *value* when ``g[j] == pos``) with the
      assignment value at *j*.  Smaller → **not canonical** (prune the
      entire branch); larger → *g* is dead (drop it); equal → continue
      scanning.
    - If ``g[j] > pos`` (unresolvable): the image depends on an
      unassigned site.  Since all earlier positions were equal, the
      verdict is deferred and *g* stays active.

    Args:
        active: Current list of non-identity group elements still
            "alive".
        assignment: Partial assignment built so far (length *pos*).
            Values at indices ``0 .. pos-1`` are set; index *pos*
            is being decided.
        pos: The site index currently being assigned.
        value: The candidate value (0 or 1) for position *pos*.

    Returns:
        ``(new_active, is_canonical)`` where *new_active* is the
        surviving active set and *is_canonical* is ``False`` if any
        element proves the current prefix is not lex-minimum.

    References:
        McKay (1998), §3: the active-set refinement is the
        incremental form of the canonical-deletion test.
    """
    new_active: list[tuple[int, ...]] = []

    for g in active:
        verdict = "active"

        for j in range(pos + 1):
            mapped = g[j]

            if mapped > pos:
                break

            img_val = assignment[mapped] if mapped < pos else value
            a_val = assignment[j] if j < pos else value

            if img_val < a_val:
                verdict = "prune"
                break
            if img_val > a_val:
                verdict = "dead"
                break

        if verdict == "prune":
            return [], False
        if verdict == "active":
            new_active.append(g)

    return new_active, True
