"""Contract tests for standalone scripts/sliced_certify.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import pytest

from archimono.certification import load_case_json, load_case_labels
from archimono.workflows.certify import (
    CertificationWorkflowConfig,
    run_certification_workflow,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "sliced_certify.py"

spec = importlib.util.spec_from_file_location("sliced_certify", _SCRIPT_PATH)
assert spec is not None
sliced_certify = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sliced_certify)


def test_sliced_certify_end_to_end(tmp_path: pathlib.Path) -> None:
    """Run plan -> run-task (all) -> reduce end-to-end and match certify output."""
    manifest_path = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"
    cert_json_dir = tmp_path / "cert_json"

    # Step 1: plan
    plan_args = argparse.Namespace(
        tilings=["hexagonal"],
        target_n=12,
        max_states_per_slice=100,
        max_slices=4,
        out=str(manifest_path),
    )
    assert sliced_certify.cmd_plan(plan_args) == 0
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    total_tasks = manifest["total_tasks"]
    assert total_tasks > 0

    # Step 2: run-task for all task_ids
    for task_id in range(total_tasks):
        run_args = argparse.Namespace(
            manifest=str(manifest_path),
            task_id=task_id,
            out_dir=str(shard_dir),
        )
        assert sliced_certify.cmd_run_task(run_args) == 0

    # Step 3: reduce and write certification output
    reduce_args = argparse.Namespace(
        manifest=str(manifest_path),
        shard_dir=str(shard_dir),
        out=str(tmp_path / "summary.json"),
        certification_json_dir=str(cert_json_dir),
    )
    assert sliced_certify.cmd_reduce(reduce_args) == 0

    # Verify JSON loadable via load_case_json
    # hexagonal target_n=12 has 9 supercell matrices that pass all three PBC
    # validity checks (self-loop, bipartite-parity, edge aliasing) --
    # confirmed directly against valid_supercell_matrices(hexagonal, 12).
    json_files = sorted(cert_json_dir.glob("*.json"))
    assert len(json_files) == 9
    for jf in json_files:
        record = load_case_json(jf)
        assert record.tiling_key == "hexagonal"
        assert record.status == "certified"
        assert record.certified_cut_value is not None

        # Load labels
        labels = load_case_labels(record, cert_json_dir)
        assert len(labels) == record.n_optimal_assignments

    # Step 4: Compare with certify workflow output
    certify_out_dir = tmp_path / "certify_out"
    certify_config = CertificationWorkflowConfig(
        solver="bruteforce",
        target_sizes=[12],
        tilings=["hexagonal"],
        one_supercell=False,
        json_out_dir=certify_out_dir,
        markdown_out=tmp_path / "report.md",
        from_json_dir=None,
        recursive=False,
        memory_limit_gb=8.0,
        bruteforce_md=tmp_path / "bf_report.md",
    )
    run_certification_workflow(certify_config)
    certify_files = sorted(certify_out_dir.glob("*.json"))
    assert len(certify_files) == 9

    # Map records by filename
    sliced_records = {jf.name: load_case_json(jf) for jf in json_files}
    certify_records = {cf.name: load_case_json(cf) for cf in certify_files}

    assert set(sliced_records.keys()) == set(certify_records.keys())

    for name, record in sliced_records.items():
        certify_record = certify_records[name]
        assert record.certified_cut_value == pytest.approx(
            certify_record.certified_cut_value
        )
        assert record.n_optimal_assignments == certify_record.n_optimal_assignments

        # Load and compare label sets
        labels_sliced = set(load_case_labels(record, cert_json_dir))
        labels_certify = set(load_case_labels(certify_record, certify_out_dir))
        assert labels_sliced == labels_certify


def test_reduce_rejects_shard_with_tampered_bounds(tmp_path: pathlib.Path) -> None:
    """Reduce recomputes [lo, hi) from slice_bounds and rejects bad metadata."""
    manifest_path = tmp_path / "manifest.json"
    shard_dir = tmp_path / "shards"

    plan_args = argparse.Namespace(
        tilings=["hexagonal"],
        target_n=12,
        max_states_per_slice=100,
        max_slices=4,
        out=str(manifest_path),
    )
    assert sliced_certify.cmd_plan(plan_args) == 0
    manifest = json.loads(manifest_path.read_text())

    for task_id in range(manifest["total_tasks"]):
        run_args = argparse.Namespace(
            manifest=str(manifest_path),
            task_id=task_id,
            out_dir=str(shard_dir),
        )
        assert sliced_certify.cmd_run_task(run_args) == 0

    # Tamper one shard: shift its window by one rank while keeping the span
    # (and thus n_states_evaluated) self-consistent, so only a slice_bounds
    # recomputation can catch it.
    case = manifest["cases"][0]
    bad_shard = shard_dir / case["case_id"] / "shard_0.json"
    shard = json.loads(bad_shard.read_text())
    shard["lo"] += 1
    shard["hi"] += 1
    bad_shard.write_text(json.dumps(shard))

    reduce_args = argparse.Namespace(
        manifest=str(manifest_path),
        shard_dir=str(shard_dir),
        out=None,
        certification_json_dir=None,
    )
    with pytest.raises(ValueError, match="slice_bounds gives"):
        sliced_certify.cmd_reduce(reduce_args)


def test_plan_cases_rejects_an_int64_overflowing_case_before_any_dispatch() -> None:
    """A case whose rank space exceeds int64 is rejected at plan time (#89)."""
    with pytest.raises(OverflowError, match="exceeds signed int64 capacity"):
        sliced_certify.plan_cases(
            ["hexagonal"],
            target_n=134,
            max_states_per_slice=1000,
            max_slices=10,
        )
