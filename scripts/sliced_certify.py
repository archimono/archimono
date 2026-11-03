"""HPC campaign runner for parallel brute-force MAX-CUT certification.

This script implements:
1. `plan`: enumerates cases and target sizes and partitions them into tasks;
2. `run-task`: executes one slice of the rank space in parallel;
3. `reduce`: merges all slice shards into exact CertificationRecord outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np

from archimono.assignment.solvers import _bruteforce, _combinatorics, _jit
from archimono.certification import (
    TILINGS,
    build_bruteforce_record,
    format_supercell,
    resolve_tilings,
    write_case_json,
)
from archimono.tilings import registry, valid_supercell_matrices

_GENERATED_BY = "archimono sliced-certify"
_TOLERANCE = 1e-9


def _iter_slice_cuts(
    lo: int,
    hi: int,
    n_nodes: int,
    n_b: int,
    score: Callable[[tuple[int, ...]], float],
) -> Iterator[tuple[int, float]]:
    """Yield ``(rank, cut)`` for every rank in ``[lo, hi)`` of the rank space.

    Unranks once to ``lo`` and advances one combination at a time, applying
    ``score`` to each candidate so the unrank/iterate logic lives in one place.

    Args:
        lo: Inclusive start rank of the slice.
        hi: Exclusive end rank of the slice.
        n_nodes: Size of the ground set.
        n_b: Subset size (number of ``1`` labels).
        score: Maps a combination to its weighted cut value.

    Yields:
        ``(rank, cut)`` pairs in ascending rank order.
    """
    combo: tuple[int, ...] | None = _combinatorics.unrank_combination(
        lo, n_nodes, n_b
    )
    r = lo
    while r < hi:
        assert combo is not None
        yield r, score(combo)
        combo = _combinatorics.next_combination(combo, n_nodes)
        r += 1


def slice_bounds(total: int, n_slices: int, slice_index: int) -> tuple[int, int]:
    """Split [0, total) rank space into non-overlapping contiguous slices."""
    if n_slices < 1:
        raise ValueError(f"n_slices must be at least 1; got {n_slices}.")
    if slice_index < 0 or slice_index >= n_slices:
        raise ValueError(
            f"slice_index must satisfy 0 <= slice_index < {n_slices}; "
            f"got {slice_index}."
        )
    base_val, remainder = divmod(total, n_slices)
    lo = slice_index * base_val + min(slice_index, remainder)
    size = base_val + (1 if slice_index < remainder else 0)
    return lo, lo + size


def auto_slice_count(
    total_states: int,
    max_states_per_slice: int,
    max_slices: int,
) -> int:
    """Return the per-case slice count for a search of total_states."""
    if max_states_per_slice < 1:
        raise ValueError(
            f"max_states_per_slice must be at least 1; got {max_states_per_slice}."
        )
    if max_slices < 1:
        raise ValueError(f"max_slices must be at least 1; got {max_slices}.")
    if total_states <= 0:
        return 1
    needed = math.ceil(total_states / max_states_per_slice)
    return max(1, min(needed, max_slices))


def plan_cases(
    tiling_keys: list[str],
    target_n: int,
    *,
    max_states_per_slice: int,
    max_slices: int,
) -> list[dict[str, Any]]:
    """Enumerate every tiling/supercell case at target_n and size slices.

    Raises:
        OverflowError: If any case's rank space exceeds signed ``int64``
            capacity -- checked here so an unsupported case is rejected
            before any HPC work is dispatched, rather than failing in each
            worker's `run-task` shard.
    """
    registry._ensure_loaded()
    cases: list[dict[str, Any]] = []
    for key in tiling_keys:
        tiling = registry.get(key)
        for matrix in valid_supercell_matrices(tiling, target_n):
            a, b, d = int(matrix[0, 0]), int(matrix[0, 1]), int(matrix[1, 1])
            realized_n = a * d * tiling.n_vertices
            _combinatorics.check_int64_rank_capacity(realized_n)
            n_b = realized_n // 2
            total_states = math.comb(realized_n, n_b)
            n_slices = auto_slice_count(
                total_states,
                max_states_per_slice,
                max_slices,
            )
            cases.append(
                {
                    "case_id": f"{key}_n{realized_n}_{a}-{b}-{d}",
                    "tiling": key,
                    "supercell": [[a, b], [0, d]],
                    "realized_n": realized_n,
                    "n_b": n_b,
                    "n_slices": n_slices,
                }
            )
    return cases


def resolve_array_task(
    cases: list[dict[str, Any]],
    task_id: int,
) -> tuple[dict[str, Any], int]:
    """Map a flat task_id to (case, slice_index)."""
    if task_id < 0:
        raise ValueError(f"task_id must be non-negative; got {task_id}.")
    offset = 0
    for case in cases:
        n_slices = int(case["n_slices"])
        if task_id < offset + n_slices:
            return case, task_id - offset
        offset += n_slices
    raise ValueError(f"task_id {task_id} out of range; only {offset} tasks exist.")


def cmd_plan(args: argparse.Namespace) -> int:
    """Run plan subcommand."""
    tiling_pairs = resolve_tilings(args.tilings, catalogue=TILINGS)
    tilings = [key for key, _ in tiling_pairs]
    cases = plan_cases(
        tilings,
        args.target_n,
        max_states_per_slice=args.max_states_per_slice,
        max_slices=args.max_slices,
    )
    total_tasks = sum(int(c["n_slices"]) for c in cases)
    manifest = {
        "target_n": args.target_n,
        "tilings": tilings,
        "total_tasks": total_tasks,
        "cases": cases,
        "generated_by": _GENERATED_BY,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"planned {len(cases)} cases -> {total_tasks} tasks "
        f"(submit --array=0-{total_tasks - 1}) -> {args.out}",
        flush=True,
    )
    return 0


def cmd_run_task(args: argparse.Namespace) -> int:
    """Run run-task subcommand."""
    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    case, slice_index = resolve_array_task(cases, args.task_id)

    tiling_key = case["tiling"]
    supercell = case["supercell"]
    n_b = int(case["n_b"])
    n_slices = int(case["n_slices"])

    # Load graph
    tiling_obj = registry.get(tiling_key)
    graph = tiling_obj.graph(np.array(supercell, dtype=np.intp))
    n_nodes = graph.number_of_nodes()

    # Form edges
    edges = [
        (int(u), int(v), float(data.get("weight", 1.0)))
        for u, v, data in graph.edges(data=True)
    ]
    total_states = math.comb(n_nodes, n_b)
    lo, hi = slice_bounds(total_states, n_slices, slice_index)

    t0 = time.perf_counter()
    best_cut: float = -math.inf
    optimal_ranks: list[int] = []
    optimal_cuts: list[float] = []
    if hi <= lo:
        pass
    elif _jit.HAS_NUMBA:
        src, dst, wt = _jit.edges_to_arrays(edges)
        binom = _combinatorics.binomial_table(n_nodes)

        # Single parallel scan over the slice: find the best cut and collect
        # every tying rank in one pass (no redundant rescan). ``optimal_cuts``
        # stays a parallel list of the tied cut value, identical in meaning to
        # before.
        best_cut_val, tie_ranks = _jit.collect_parallel_best_and_ties(
            lo, hi, n_nodes, n_b, binom, src, dst, wt
        )
        best_cut = float(best_cut_val)
        optimal_ranks = [int(rank) for rank in tie_ranks]
        optimal_cuts = [best_cut] * len(optimal_ranks)
    else:
        # Serial Python fallback: score the whole slice once, then keep the ties.
        def score_py(combo: tuple[int, ...]) -> float:
            labels = _bruteforce.labels_from_one_nodes(
                n_nodes=n_nodes, one_nodes=combo
            )
            return float(sum(wt for u, v, wt in edges if labels[u] != labels[v]))

        scored = list(_iter_slice_cuts(lo, hi, n_nodes, n_b, score_py))
        best_cut = max((cut for _, cut in scored), default=-math.inf)
        for rank, cut in scored:
            if math.isclose(cut, best_cut, rel_tol=0.0, abs_tol=_TOLERANCE):
                optimal_ranks.append(rank)
                optimal_cuts.append(cut)

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    shard = {
        "tiling": tiling_key,
        "supercell": supercell,
        "n_nodes": n_nodes,
        "n_b": n_b,
        "slice_index": slice_index,
        "n_slices": n_slices,
        "lo": lo,
        "hi": hi,
        "best_cut": best_cut if math.isfinite(best_cut) else None,
        "optimal_ranks": optimal_ranks,
        "optimal_cuts": optimal_cuts,
        "n_states_evaluated": hi - lo,
        "runtime_ms": runtime_ms,
        "generated_by": _GENERATED_BY,
    }

    out_dir = pathlib.Path(args.out_dir)
    case_dir = out_dir / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    shard_path = case_dir / f"shard_{slice_index}.json"
    shard_path.write_text(json.dumps(shard, indent=2))

    print(
        f"task {args.task_id} -> {shard_path} best_cut={best_cut} "
        f"runtime={runtime_ms / 1000.0:.3f}s",
        flush=True,
    )
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    """Run reduce subcommand."""
    manifest_path = pathlib.Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]

    shard_dir = pathlib.Path(args.shard_dir)
    certification_json_dir = (
        pathlib.Path(args.certification_json_dir)
        if args.certification_json_dir
        else None
    )

    reduced_cases: list[dict[str, Any]] = []
    t0_global = time.perf_counter()

    for case in cases:
        case_id = case["case_id"]
        tiling_key = case["tiling"]
        supercell = case["supercell"]
        n_b = int(case["n_b"])
        n_slices = int(case["n_slices"])

        # Check shards
        shards: list[dict[str, Any]] = []
        for idx in range(n_slices):
            shard_path = shard_dir / case_id / f"shard_{idx}.json"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing expected shard: {shard_path}")
            shards.append(json.loads(shard_path.read_text()))

        slice_runtimes_ms = [float(s["runtime_ms"]) for s in shards]
        n_nodes = int(shards[0]["n_nodes"])
        expected_states = math.comb(n_nodes, n_b)

        # Partition verification: recompute each slice's [lo, hi) from
        # slice_bounds rather than trusting the bounds the worker recorded, so
        # misaligned or overlapping shards cannot pass even if their evaluated
        # counts happen to sum correctly.
        for idx, s in enumerate(shards):
            lo, hi = slice_bounds(expected_states, n_slices, idx)
            if int(s["lo"]) != lo or int(s["hi"]) != hi:
                raise ValueError(
                    f"Shard {idx} of case {case_id} reports bounds "
                    f"[{s['lo']}, {s['hi']}) but slice_bounds gives [{lo}, {hi})."
                )
            if int(s["n_states_evaluated"]) != hi - lo:
                raise ValueError(
                    f"Shard {idx} of case {case_id} evaluated "
                    f"{s['n_states_evaluated']} states but its slice spans "
                    f"{hi - lo}."
                )

        total_states = sum(int(s["n_states_evaluated"]) for s in shards)
        if total_states != expected_states:
            raise ValueError(
                f"Evaluated states {total_states} != expected {expected_states} "
                f"for case {case_id}."
            )

        # Reduce
        best_cut = max(
            (-math.inf if s["best_cut"] is None else float(s["best_cut"]))
            for s in shards
        )
        keep_ranks: set[int] = set()
        for s in shards:
            for rank, cut in zip(s["optimal_ranks"], s["optimal_cuts"]):
                if math.isclose(cut, best_cut, rel_tol=0.0, abs_tol=_TOLERANCE):
                    keep_ranks.add(int(rank))

        optimal_ranks = sorted(keep_ranks)

        # Convert to optimal labels
        optimal_labels: list[tuple[int, ...]] = []
        for rank in optimal_ranks:
            combo = _combinatorics.unrank_combination(rank, n_nodes, n_b)
            labels = _bruteforce.labels_from_one_nodes(
                n_nodes=n_nodes, one_nodes=combo
            )
            optimal_labels.append(tuple(labels))

        # Write certification JSON
        if certification_json_dir is not None:
            tiling_obj = registry.get(tiling_key)
            display_lookup = dict(TILINGS)
            tiling_display = display_lookup.get(tiling_key, tiling_key)
            vertex_config = tiling_obj.vertex_config
            n_edges = tiling_obj.graph(
                np.array(supercell, dtype=np.intp)
            ).number_of_edges()

            # Format supercell string
            supercell_str = format_supercell(np.array(supercell))

            rec = build_bruteforce_record(
                tiling_key=tiling_key,
                tiling_display=tiling_display,
                vertex_config=vertex_config,
                target_n=int(manifest["target_n"]),
                realized_n=n_nodes,
                supercell=supercell_str,
                n_edges=n_edges,
                cut_value=best_cut,
                runtime_ms=sum(slice_runtimes_ms),
                target_composition=(n_nodes - n_b, n_b),
                optimal_labels=optimal_labels,
                total_states=expected_states,
                solver_metadata={
                    "max_slice_runtime_ms": max(slice_runtimes_ms, default=0.0),
                    "timed_slice_count": len(slice_runtimes_ms),
                },
            )
            write_case_json(
                out_dir=certification_json_dir,
                record=rec,
                generated_by=_GENERATED_BY,
            )

        reduced_cases.append(
            {
                "case_id": case_id,
                "tiling": tiling_key,
                "supercell": supercell,
                "n_nodes": n_nodes,
                "n_b": n_b,
                "best_cut": best_cut,
                "n_optimal_assignments": len(optimal_labels),
                "optimal_labels": [list(lbl) for lbl in optimal_labels],
                "n_states_evaluated": total_states,
                "runtime_ms": sum(slice_runtimes_ms),
                "complete": True,
            }
        )

    duration = time.perf_counter() - t0_global

    if certification_json_dir is not None:
        print(
            f"wrote {len(reduced_cases)} certification JSON files to "
            f"{certification_json_dir}",
            flush=True,
        )

    if args.out is not None:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "target_n": manifest["target_n"],
            "tilings": manifest["tilings"],
            "runtime_s": duration,
            "results": reduced_cases,
            "generated_by": _GENERATED_BY,
        }
        out_path.write_text(json.dumps(summary, indent=2))
        print(
            f"reduced {len(reduced_cases)} cases runtime={duration:.3f}s -> {args.out}",
            flush=True,
        )
    else:
        print(
            f"reduced {len(reduced_cases)} cases runtime={duration:.3f}s",
            flush=True,
        )
    return 0


def main() -> int:
    """Top-level entry point."""
    parser = argparse.ArgumentParser(prog="sliced_certify")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # plan
    plan_p = subparsers.add_parser("plan")
    plan_p.add_argument("--tilings", nargs="+", default=["all"])
    plan_p.add_argument("--target-n", type=int, required=True)
    plan_p.add_argument("--max-states-per-slice", type=int, default=1000000000)
    plan_p.add_argument("--max-slices", type=int, default=64)
    plan_p.add_argument("--out", required=True)
    plan_p.set_defaults(func=cmd_plan)

    # run-task
    run_p = subparsers.add_parser("run-task")
    run_p.add_argument("--manifest", required=True)
    run_p.add_argument("--task-id", type=int, required=True)
    run_p.add_argument("--out-dir", required=True)
    run_p.set_defaults(func=cmd_run_task)

    # reduce
    red_p = subparsers.add_parser("reduce")
    red_p.add_argument("--manifest", required=True)
    red_p.add_argument("--shard-dir", required=True)
    red_p.add_argument("--out")
    red_p.add_argument("--certification-json-dir")
    red_p.set_defaults(func=cmd_reduce)

    args = parser.parse_args()
    return cast(int, args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
