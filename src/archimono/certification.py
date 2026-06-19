"""Certification record schema, I/O helpers, and stress-test utilities.

This module defines the per-case JSON schema used by stress-case certification
scripts and provides helpers to write, reload, and summarise those records.
It also exposes shared constants and helpers used across all certification
scripts.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Tiling catalogue
# ---------------------------------------------------------------------------

TILINGS: list[tuple[str, str]] = [
    ("hexagonal", "6³ (hexagonal)"),
    ("square", "4⁴ (square)"),
    ("truncated_square", "4.8² (truncated square)"),
    ("truncated_trihexagonal", "4.6.12 (truncated trihexagonal)"),
    ("triangular", "3⁶ (triangular)"),
    ("kagome", "3.6.3.6 (kagome)"),
    ("truncated_hexagonal", "3.12.12 (truncated hexagonal)"),
    ("elongated_triangular", "3³.4² (elongated triangular)"),
    ("rhombitrihexagonal", "3.4.6.4 (rhombitrihexagonal)"),
    ("snub_square", "3².4.3.4 (snub square)"),
    ("snub_hexagonal", "3⁴.6 (snub hexagonal)"),
]

FRUSTRATED_TILING_KEYS: frozenset[str] = frozenset(
    {
        "triangular",
        "kagome",
        "truncated_hexagonal",
        "elongated_triangular",
        "rhombitrihexagonal",
        "snub_square",
        "snub_hexagonal",
    }
)

DEFAULT_TILING_ORDER: list[str] = [key for key, _ in TILINGS]

_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SIDECAR_ENCODING: str = "np-packbits-v1"


# ---------------------------------------------------------------------------
# CertificationRecord
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class CertificationRecord:
    """Exact certification result for one ``(tiling, target_n)`` case."""

    tiling_key: str
    tiling_display: str
    solver: str
    assignment_index: int
    n_optimal_assignments: int
    vertex_config: str
    target_n: int
    realized_n: int | None
    supercell: str | None
    n_edges: int | None
    certified_cut_value: float | None
    certified_cut_efficiency: float | None
    runtime_ms: float
    status: str
    exact_planar_cut_value: float | None = None
    note: str = ""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# New shared helpers
# ---------------------------------------------------------------------------


def format_supercell(matrix: npt.NDArray[np.intp]) -> str:
    """Format a 2×2 HNF supercell matrix as a compact string.

    Diagonal matrices (zero upper-right entry) are formatted as ``(axb)``.
    Shear matrices (non-zero upper-right entry) are formatted as
    ``[a b; 0 d]`` to preserve all four entries.

    Args:
        matrix: A 2×2 integer array in Hermite Normal Form, with
            ``matrix[0, 0]`` and ``matrix[1, 1]`` as diagonal entries and
            ``matrix[0, 1]`` as the upper-right shear entry.

    Returns:
        A human-readable string representation of the supercell.
    """
    a = int(matrix[0, 0])
    b_shear = int(matrix[0, 1])
    d = int(matrix[1, 1])
    if b_shear == 0:
        return f"({a}x{d})"
    return f"[{a} {b_shear}; 0 {d}]"


def resolve_target_sizes(
    target_sizes: list[int] | None,
    *,
    default: list[int],
    max_n: int,
) -> list[int]:
    """Return sorted positive target sizes from an explicit list or filtered defaults.

    When ``target_sizes`` is ``None`` the defaults are used, filtered to
    positive values ``<= max_n`` and deduplicated. When an explicit list is
    given it is filtered to positive values only (max_n is ignored) and
    deduplicated.

    Args:
        target_sizes: Explicit sizes to use, or ``None`` to fall back to
            ``default``.
        default: Default size list used when ``target_sizes`` is ``None``.
        max_n: Upper bound applied to ``default`` when ``target_sizes``
            is ``None``.

    Returns:
        Sorted, deduplicated list of positive integers.
    """
    if target_sizes is None:
        return sorted(set(n for n in default if 0 < n <= max_n))
    return sorted(set(n for n in target_sizes if n > 0))


def resolve_tilings(
    keys: list[str] | None,
    *,
    catalogue: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Expand tiling key specifiers to ``(registry_key, display)`` pairs.

    Accepts the special tokens ``"all"`` and ``"frustrated"`` as well as
    individual registry keys. Matching is case-insensitive. ``None`` is
    treated as ``["all"]``.

    Args:
        keys: List of tiling keys (or special tokens ``"all"`` /
            ``"frustrated"``).  Pass ``None`` to request all tilings.
        catalogue: Full tiling catalogue to resolve against.  Defaults to
            :data:`TILINGS` when ``None``.

    Returns:
        Ordered list of ``(registry_key, display_name)`` pairs.

    Raises:
        ValueError: If any key is not a recognised registry key or special
            token.
    """
    source = catalogue if catalogue is not None else TILINGS
    if keys is None:
        return list(source)
    key_to_display: dict[str, tuple[str, str]] = {
        k.lower(): (k, d) for k, d in source
    }
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in keys:
        normalized = token.lower()
        if normalized == "all":
            for k, d in source:
                if k not in seen:
                    result.append((k, d))
                    seen.add(k)
        elif normalized == "frustrated":
            for k, d in source:
                if k in FRUSTRATED_TILING_KEYS and k not in seen:
                    result.append((k, d))
                    seen.add(k)
        elif normalized in key_to_display:
            key, display = key_to_display[normalized]
            if key not in seen:
                result.append((key, display))
                seen.add(key)
        else:
            raise ValueError(
                f"unknown tiling key or token: {token!r}. "
                f"Valid keys: {sorted(k for k, _ in source)} "
                "plus 'all', 'frustrated'."
            )
    return result


def make_skip_record(
    *,
    tiling_key: str,
    tiling_display: str,
    solver: str,
    target_n: int,
    vertex_config: str,
    note: str = "no valid supercell",
) -> CertificationRecord:
    """Factory for ``status='skip'`` placeholder records.

    Creates a :class:`CertificationRecord` for cases where no valid supercell
    exists or the case should otherwise be skipped, with all nullable result
    fields set to ``None`` and ``runtime_ms`` set to ``0.0``.

    Args:
        tiling_key: Registry key of the tiling.
        tiling_display: Human-readable display name of the tiling.
        solver: Name of the solver that would have been used.
        target_n: Target number of vertices requested.
        vertex_config: Vertex composition string (e.g. ``"AB"``).
        note: Human-readable explanation for the skip.

    Returns:
        A :class:`CertificationRecord` with ``status='skip'``.
    """
    return CertificationRecord(
        tiling_key=tiling_key,
        tiling_display=tiling_display,
        solver=solver,
        assignment_index=0,
        n_optimal_assignments=0,
        vertex_config=vertex_config,
        target_n=target_n,
        realized_n=None,
        supercell=None,
        n_edges=None,
        certified_cut_value=None,
        certified_cut_efficiency=None,
        runtime_ms=0.0,
        status="skip",
        note=note,
    )


# ---------------------------------------------------------------------------
# Packed-bits sidecar I/O (private)
# ---------------------------------------------------------------------------


def _save_packed_labels(
    path: pathlib.Path,
    labels: list[tuple[int, ...]],
    label_width: int,
) -> None:
    """Write binary label tuples to a compressed NumPy packed-bits file.

    Args:
        path: Destination ``.npz`` file path.
        labels: List of label tuples, each of length ``label_width``.
        label_width: Number of bits per label.

    Raises:
        ValueError: If ``labels`` is empty or ``label_width`` is zero.
    """
    if not labels or label_width == 0:
        raise ValueError("labels must be non-empty and label_width must be positive.")
    n = len(labels)
    # Pad each row to a multiple of 8 so np.packbits works cleanly along axis=1.
    row_width = ((label_width + 7) // 8) * 8
    arr = np.zeros((n, row_width), dtype=np.uint8)
    arr[:, :label_width] = np.array(labels, dtype=np.uint8)
    packed: npt.NDArray[np.uint8] = np.packbits(arr, axis=1, bitorder="big")
    np.savez_compressed(
        path,
        labels_packed=packed,
        width=np.array(label_width, dtype=np.uint16),
        count=np.array(n, dtype=np.uint64),
    )


def _load_packed_labels(path: pathlib.Path) -> list[tuple[int, ...]]:
    """Load binary label tuples from a packed-bits NumPy sidecar file.

    Args:
        path: Path to a ``.npz`` file written by :func:`_save_packed_labels`.

    Returns:
        List of label tuples with the original bit-width.

    Raises:
        FileNotFoundError: If the sidecar file does not exist.
        ValueError: If the stored count does not match the packed array shape.
    """
    data = np.load(path)
    labels_packed = np.asarray(data["labels_packed"], dtype=np.uint8)
    width = int(data["width"])
    count = int(data["count"])
    if labels_packed.shape[0] != count:
        raise ValueError(
            f"Sidecar count mismatch: expected {count}, got {labels_packed.shape[0]}."
        )
    unpacked = np.unpackbits(labels_packed, axis=1, bitorder="big")
    bits = unpacked[:, :width]
    return [tuple(int(b) for b in row) for row in bits]


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def case_json_name(record: CertificationRecord) -> str:
    """Return the canonical JSON filename for one certification record.

    Exactly one record (one representative) is written per
    ``(tiling, supercell, target_n)`` case; every optimal assignment for that
    case is stored in the companion ``-labels.npz`` sidecar, so no per-optimum
    ``-optN`` suffix is appended. The distinct supercell slug keeps filenames
    unique. ``assignment_index`` is retained in the schema but no longer used
    in the filename.
    """
    supercell_slug = _supercell_slug(record.supercell)
    return f"{record.tiling_key}-n{record.target_n}-{supercell_slug}.json"


def write_case_json(
    *,
    out_dir: pathlib.Path,
    record: CertificationRecord,
    generated_by: str,
) -> pathlib.Path:
    """Write one per-case certification JSON file.

    When ``record.metadata`` contains an ``"all_labels"`` key the label
    assignments are written to a companion ``.npz`` sidecar in ``out_dir``
    and the JSON stores only a compact manifest in its place.

    Args:
        out_dir: Destination directory for per-case JSON files.
        record: Certification record to serialize.
        generated_by: Script or module name producing this record.

    Returns:
        The path written to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / case_json_name(record)
    record_dict = dataclasses.asdict(record)

    all_labels: list[Any] | None = record_dict["metadata"].get("all_labels")
    if all_labels:
        label_width = len(all_labels[0])
        sidecar_name = path.stem + "-labels.npz"
        _save_packed_labels(
            out_dir / sidecar_name,
            [tuple(int(v) for v in lbl) for lbl in all_labels],
            label_width,
        )
        del record_dict["metadata"]["all_labels"]
        record_dict["metadata"]["label_encoding"] = _SIDECAR_ENCODING
        record_dict["metadata"]["label_width"] = label_width
        record_dict["metadata"]["label_count"] = len(all_labels)
        record_dict["metadata"]["labels_file"] = sidecar_name

    payload = {
        "generated_by": generated_by,
        "record": record_dict,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_case_json(path: pathlib.Path) -> CertificationRecord:
    """Load one per-case certification JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CertificationRecord(**payload["record"])


def load_case_json_dir(out_dir: pathlib.Path) -> list[CertificationRecord]:
    """Load every per-case certification JSON file from a directory."""
    records = [load_case_json(path) for path in sorted(out_dir.glob("*.json"))]
    return sorted(records, key=_record_sort_key)


def load_case_json_tree(root_dir: pathlib.Path) -> list[CertificationRecord]:
    """Load every per-case certification JSON file from a directory tree."""
    records = [load_case_json(path) for path in sorted(root_dir.rglob("*.json"))]
    return sorted(records, key=_record_sort_key)


def load_case_labels(
    record: CertificationRecord,
    json_dir: pathlib.Path,
) -> list[tuple[int, ...]]:
    """Load the optimal label assignments for a certification record.

    Handles both the sidecar format (``label_encoding`` manifest in
    ``record.metadata``) and the legacy inline format (``all_labels`` list).
    :func:`load_case_json` does not auto-load the sidecar, so markdown
    generation remains fast.

    Args:
        record: A certification record returned by :func:`load_case_json`.
        json_dir: Directory that contains the companion ``.npz`` sidecar, if any.

    Returns:
        All optimal label assignments as a list of integer tuples. Returns an
        empty list if the record contains no label data.

    Raises:
        FileNotFoundError: If the referenced sidecar file does not exist.
        ValueError: If an unknown ``label_encoding`` value is found.
    """
    encoding: str | None = record.metadata.get("label_encoding")
    if encoding is None:
        raw: list[Any] = record.metadata.get("all_labels", [])
        return [tuple(int(b) for b in lbl) for lbl in raw]
    if encoding == _SIDECAR_ENCODING:
        return _load_packed_labels(json_dir / record.metadata["labels_file"])
    raise ValueError(f"Unknown label encoding: {encoding!r}")


# ---------------------------------------------------------------------------
# Markdown report rendering
# ---------------------------------------------------------------------------

_DEFAULT_DESCRIPTION = (
    "Fixed-composition MAX-CUT values in this report are independently certified."
)

_DEFAULT_TITLE = "Stress-Test Corpus Results"


def render_markdown_summary(
    records: list[CertificationRecord],
    *,
    title: str = _DEFAULT_TITLE,
    description: str = _DEFAULT_DESCRIPTION,
    extra_columns: list[tuple[str, Callable[[CertificationRecord], str]]] | None = None,
) -> str:
    """Render certification records as a markdown summary table.

    Args:
        records: Certification results loaded from per-case JSON files.
        title: Report title inserted as the top-level heading.
        description: One-sentence description of the solver or method that
            produced these results, inserted below the report title.
        extra_columns: Optional solver-specific columns to append after the
            fixed columns. Each entry is a ``(header, extractor)`` pair where
            ``extractor`` receives a :class:`CertificationRecord` and returns
            the formatted cell string.

    Returns:
        Markdown text in the same report shape as the certification script.
    """
    extra_columns = extra_columns or []
    case_records = _collapse_case_records(records)
    target_sizes = sorted({record.target_n for record in case_records})
    seen_keys: set[str] = set()
    tiling_displays: list[str] = []
    for key in DEFAULT_TILING_ORDER:
        if key not in seen_keys and any(r.tiling_key == key for r in case_records):
            display = next(
                r.tiling_display for r in case_records if r.tiling_key == key
            )
            tiling_displays.append(_markdown_tiling_display(display))
            seen_keys.add(key)
    for r in case_records:
        if r.tiling_key not in seen_keys:
            tiling_displays.append(_markdown_tiling_display(r.tiling_display))
            seen_keys.add(r.tiling_key)
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Scope",
        "",
        f"- Target sizes: {target_sizes}",
        f"- Tilings ({len(tiling_displays)}):",
        "",
    ] + [f"  - {t}" for t in tiling_displays] + [
        "",
        "## Certification summary",
        "",
    ]

    records_by_tiling: dict[str, list[CertificationRecord]] = {}
    for record in sorted(case_records, key=_record_sort_key):
        records_by_tiling.setdefault(record.tiling_display, []).append(record)

    extra_headers = "".join(f" {h} |" for h, _ in extra_columns)
    extra_sep = "".join(" ---: |" for _ in extra_columns)
    for tiling_display, tiling_records in records_by_tiling.items():
        lines += [
            f"### {_markdown_tiling_display(tiling_display)}",
            "",
            f"| n | supercell | k* | cut efficiency | runtime (ms) |{extra_headers}",
            f"|---:|---|---:|---:|---:|{extra_sep}",
        ]
        for record in tiling_records:
            certified_cut = (
                "-"
                if record.certified_cut_value is None
                else f"{record.certified_cut_value:.1f}"
            )
            certified_eff = (
                "-"
                if record.certified_cut_efficiency is None
                else f"{record.certified_cut_efficiency:.4f}"
            )
            extra_cells = "".join(f" {fn(record)} |" for _, fn in extra_columns)
            n = record.realized_n or record.target_n
            sup = record.supercell or "-"
            lines.append(
                f"| {n} | {sup} | {certified_cut} | {certified_eff}"
                f" | {record.runtime_ms:.1f} |{extra_cells}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_markdown_summary(
    path: pathlib.Path,
    records: list[CertificationRecord],
    *,
    title: str = _DEFAULT_TITLE,
    description: str = _DEFAULT_DESCRIPTION,
    extra_columns: list[tuple[str, Callable[[CertificationRecord], str]]] | None = None,
) -> None:
    """Write a markdown summary generated from per-case JSON records.

    Args:
        path: Destination file path for the markdown output.
        records: Certification results loaded from per-case JSON files.
        title: Report title, forwarded to :func:`render_markdown_summary`.
        description: One-sentence description of the solver or method that
            produced these results, forwarded to :func:`render_markdown_summary`.
        extra_columns: Forwarded to :func:`render_markdown_summary`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_summary(
            records, title=title, description=description, extra_columns=extra_columns
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Private sort/collapse helpers
# ---------------------------------------------------------------------------


def _record_sort_key(record: CertificationRecord) -> tuple[int, int, str]:
    """Return a stable sort key for report rows."""
    try:
        tiling_index = DEFAULT_TILING_ORDER.index(record.tiling_key)
    except ValueError:
        tiling_index = len(DEFAULT_TILING_ORDER)
    return tiling_index, record.target_n, record.tiling_display


def _collapse_case_records(
    records: list[CertificationRecord],
) -> list[CertificationRecord]:
    """Return one record per (tiling_key, supercell, target_n, solver), sorted by tiling order."""  # noqa: E501
    seen: dict[tuple[str, str | None, int, str], CertificationRecord] = {}
    for record in sorted(records, key=_record_sort_key):
        key = (record.tiling_key, record.supercell, record.target_n, record.solver)
        if key not in seen:
            seen[key] = record
    return list(seen.values())


def _markdown_tiling_display(tiling_display: str) -> str:
    """Convert ASCII caret exponents in a tiling label into Unicode superscripts."""
    chars: list[str] = []
    index = 0
    while index < len(tiling_display):
        if tiling_display[index] == "^" and index + 1 < len(tiling_display):
            digit_start = index + 1
            digit_end = digit_start
            while (
                digit_end < len(tiling_display) and tiling_display[digit_end].isdigit()
            ):
                digit_end += 1
            if digit_end > digit_start:
                chars.append(
                    tiling_display[digit_start:digit_end].translate(_SUPERSCRIPT_DIGITS)
                )
                index = digit_end
                continue
        chars.append(tiling_display[index])
        index += 1
    return "".join(chars)


def _supercell_slug(supercell: str | None) -> str:
    """Convert a supercell string into a filename-safe slug.

    Diagonal ``(NaxNb)`` → ``NaxNb``.
    Shear ``[a b; 0 d]`` → ``a-b_0-d`` (preserves all four entries).
    """
    if supercell is None:
        return "nosupercell"
    # Shear format: [a b; 0 d]
    if supercell.startswith("["):
        return (
            supercell.replace("[", "")
            .replace("]", "")
            .replace("; ", "_")
            .replace(" ", "-")
        )
    # Diagonal format: (NaxNb)
    return supercell.replace("(", "").replace(")", "")
