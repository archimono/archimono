"""Registry mapping vertex-configuration strings to Tiling subclasses.

Usage::

    from archimono.tilings import registry

    tiling = registry.get("3.6.3.6")   # KagomeTiling instance
    tiling = registry.get("6^3")       # HexagonalTiling instance (alias)
"""

from __future__ import annotations

from collections.abc import Callable

from archimono.tilings.base import Tiling

# Populated by each tiling module via :func:`register`.
_REGISTRY: dict[str, type[Tiling]] = {}


def register(*keys: str) -> Callable[[type[Tiling]], type[Tiling]]:
    """Class decorator that registers a :class:`Tiling` subclass under *keys*.

    Args:
        *keys: One or more vertex-configuration strings that should resolve to
            the decorated class.  At least one key is required.

    Returns:
        The unmodified class, enabling decorator stacking.

    Raises:
        ValueError: If a key is already registered to a different class.
    """

    def decorator(cls: type[Tiling]) -> type[Tiling]:
        for key in keys:
            if key in _REGISTRY and _REGISTRY[key] is not cls:
                raise ValueError(
                    f"Key '{key}' is already registered to {_REGISTRY[key].__name__}."
                )
            _REGISTRY[key] = cls
        return cls

    return decorator


def get(vertex_config: str, bond_length: float = 1.0) -> Tiling:
    """Return a :class:`Tiling` instance for the given vertex-configuration string.

    Triggers lazy import of all tiling submodules on first call so that
    callers need not import individual tiling modules themselves.

    Args:
        vertex_config: A vertex-configuration string such as ``'3.6.3.6'``
            or ``'6^3'``.
        bond_length: Edge length scale passed to the tiling constructor.
            Default ``1.0`` (dimensionless unit bond).

    Returns:
        A freshly constructed :class:`Tiling` instance.

    Raises:
        KeyError: If *vertex_config* is not recognised.
        ValueError: If *bond_length* is not positive.
    """
    _ensure_loaded()
    if vertex_config not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise KeyError(
            f"Unknown vertex config '{vertex_config}'. "
            f"Available: {available}"
        )
    return _REGISTRY[vertex_config](bond_length=bond_length)


def available() -> list[str]:
    """Return a sorted list of all registered vertex-configuration keys."""
    _ensure_loaded()
    return sorted(_REGISTRY)


_loaded: bool = False


def _ensure_loaded() -> None:
    """Import all tiling submodules so their ``@register`` decorators run."""
    global _loaded  # noqa: PLW0603
    if _loaded:
        return
    # Bipartite tilings
    import archimono.tilings._bipartite.hexagonal  # noqa: F401
    import archimono.tilings._bipartite.square  # noqa: F401
    import archimono.tilings._bipartite.truncated_square  # noqa: F401
    import archimono.tilings._bipartite.truncated_trihexagonal  # noqa: F401

    # Frustrated tilings
    import archimono.tilings._frustrated.elongated_triangular  # noqa: F401
    import archimono.tilings._frustrated.kagome  # noqa: F401
    import archimono.tilings._frustrated.rhombitrihexagonal  # noqa: F401
    import archimono.tilings._frustrated.snub_hexagonal  # noqa: F401
    import archimono.tilings._frustrated.snub_square  # noqa: F401
    import archimono.tilings._frustrated.triangular  # noqa: F401
    import archimono.tilings._frustrated.truncated_hexagonal  # noqa: F401
    _loaded = True
