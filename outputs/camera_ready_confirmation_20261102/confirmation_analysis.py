"""Frozen complete-cohort summaries and paired conditional-noise inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# isort: off
from confirmation_context import (
    ROOT,
    cap_cpu,
    case_key,
    file_hash,
    initialize_worker,
    read_json,
    verify,
    write_json,
)
from confirmation_audit import METRICS, audit_record, close
from headroom_report import episode_metrics
import numpy as np
import pandas as pd
# isort: on


def noise_indices(cells, replicates, seed, panel_tag):
    dates = sorted({row["start_day"] for row in cells})
    seeds = {
        day: sorted({row["seed"] for row in cells if row["start_day"] == day})
        for day in dates
    }
    sizes = {len(values) for values in seeds.values()}
    if len(sizes) != 1:
        raise ValueError("The registered conditional-noise panel must be balanced")
    size = next(iter(sizes))
    if (
        len({value for values in seeds.values() for value in values})
        != len(dates) * size
    ):
        raise ValueError("New confirmation streams must be date-unique")
    rng = np.random.default_rng(np.random.SeedSequence([seed, panel_tag]))
    indices = rng.integers(0, size, size=(replicates, len(dates), size))
    return dates, seeds, indices


def paired_estimate(differences, indices):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or values.shape != indices.shape[1:]:
        raise ValueError("Paired contrasts must be date by noise replicate")
    if not np.isfinite(values).all():
        raise ValueError("Never bootstrap incomplete or non-finite cohorts")
    resampled = values[np.arange(values.shape[0])[None, :, None], indices]
    means = resampled.mean(axis=(1, 2))
    date_means = values.mean(axis=1)
    leave_one_out = [
        float(np.delete(date_means, index).mean()) for index in range(len(date_means))
    ]
    return {
        "mean": float(values.mean()),
        "paired_case_sd": float(values.std(ddof=1)),
        "bootstrap_standard_error": float(means.std(ddof=1)),
        "lower_025": float(np.quantile(means, 0.025)),
        "upper_975": float(np.quantile(means, 0.975)),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "positive_cases": int((values > 0).sum()),
        "date_means": date_means.tolist(),
        "leave_one_date_out_means": leave_one_out,
    }, means


def point_frontier(rows):
    valid = {family: info for family, info in rows.items() if info["all_qualified"]}
    frontier = []
    for family, row in valid.items():
        target = np.array(
            [
                row["metrics"]["stress_pcc_relief_mw"]["mean"],
                row["metrics"]["cashflow_26h_usd"]["mean"],
            ]
        )
        dominated = False
        for other, candidate in valid.items():
            if other == family:
                continue
            values = np.array(
                [
                    candidate["metrics"]["stress_pcc_relief_mw"]["mean"],
                    candidate["metrics"]["cashflow_26h_usd"]["mean"],
                ]
            )
            if np.all(values >= target) and np.any(values > target):
                dominated = True
                break
        if not dominated:
            frontier.append(family)
    return sorted(frontier)


def read_inventory(protocol, panel):
    output = ROOT / "evaluation" / panel
    manifest = read_json(output / "manifest.json")
    status = read_json(output / "status.json")
    if (
        manifest["evaluation_registration_id"] != protocol["registration_id"]
        or manifest["cells"] != protocol["panels"][panel]["cells"]
        or status["runtime_failures"]
        or not status["finished"]
        or status["completed"] != manifest["expected_episodes"]
    ):
        raise ValueError("Evaluation inventory is incomplete or has the wrong identity")
    log = output / "completion.jsonl"
    if file_hash(log) != status["completion_log_sha256"]:
        raise ValueError("Completion receipts changed")
    with log.open() as handle:
        inventory = [json.loads(line) for line in handle]
    expected = {
        (policy["id"], case_key(cell))
        for policy in manifest["policies"]
        for cell in manifest["cells"]
    }
    observed = {(row["policy_id"], case_key(row["cell"])) for row in inventory}
    if observed != expected or len(inventory) != len(expected):
        raise ValueError("Missing or duplicate completed slots")
    for row in inventory:
        row["reused"] = False
    for row in protocol["reused_neural_evidence"].get(panel, []):
        inventory.append({**row, "panel": panel, "reused": True})
    return inventory


def audit_slot(protocol, panel, artifact):
    path = Path(artifact["path"])
    if file_hash(path) != artifact["sha256"]:
        raise ValueError("Episode changed after its completion receipt")
    record = read_json(path)
    policy = next(
        row for row in protocol["policies"] if row["id"] == artifact["policy_id"]
    )
    if record["cell"] != artifact["cell"]:
        raise ValueError("Episode has the wrong registered condition")
    if artifact["reused"]:
        if (
            record["policy"]["checkpoint"]["sha256"] != policy["checkpoint"]["sha256"]
            or record["policy"]["id"] != artifact["source_policy_id"]
        ):
            raise ValueError("Reused episode has the wrong frozen model identity")
    elif (
        record["policy"] != policy
        or record["evaluation_registration_id"] != protocol["registration_id"]
        or record["panel"] != panel
    ):
        raise ValueError("New episode has the wrong policy or study identity")
    audit = audit_record(
        path.parent,
        record,
        tracking_floor_kw=protocol["macro_tracking_floor_kw"],
    )
    return {
        "panel": panel,
        "policy_id": policy["id"],
        "family": policy["family"],
        "training_seed": policy["training_seed"],
        "kind": policy["kind"],
        "checkpoint_sha256": (
            policy["checkpoint"]["sha256"] if policy["kind"] == "neural" else None
        ),
        "cell": artifact["cell"],
        "episode_path": str(path),
        "episode_sha256": artifact["sha256"],
        "reused": artifact["reused"],
        **audit,
    }


def collect_panels(protocol, panels):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for cpu in protocol["cpus"]:
        queue.put(cpu)
    records = []
    with ProcessPoolExecutor(
        max_workers=len(protocol["cpus"]),
        mp_context=context,
        initializer=initialize_worker,
        initargs=(queue,),
    ) as pool:
        for panel in panels:
            inventory = read_inventory(protocol, panel)
            futures = [
                pool.submit(audit_slot, protocol, panel, artifact)
                for artifact in inventory
            ]
            panel_records = []
            for index, future in enumerate(as_completed(futures), 1):
                panel_records.append(future.result())
                if index % 100 == 0:
                    print(f"Raw audit {panel}: {index}/{len(inventory)}", flush=True)
            panel_records.sort(
                key=lambda row: (row["policy_id"], case_key(row["cell"]))
            )
            expected = {
                (policy["id"], case_key(cell))
                for policy in protocol["policies"]
                for cell in protocol["panels"][panel]["cells"]
            }
            observed = {
                (row["policy_id"], case_key(row["cell"])) for row in panel_records
            }
            if observed != expected or len(observed) != len(panel_records):
                raise ValueError("The full fixed-model cohort is incomplete")
            paired_inputs = defaultdict(set)
            paired_masks = defaultdict(set)
            for row in panel_records:
                if row["complete"]:
                    key = case_key(row["cell"])
                    paired_inputs[key].add(row["inputs_sha256"])
                    paired_masks[key].add(row["stress_mask_sha256"])
            if any(
                len(values) != 1
                for values in (*paired_inputs.values(), *paired_masks.values())
            ):
                raise AssertionError(
                    "Paired policies received different exogenous/reference inputs"
                )
            records.extend(panel_records)
    queue.close()
    queue.join_thread()
    return records


def summarize(protocol, records, panels):
    output = {}
    for panel in panels:
        output[panel] = {}
        for bias in (0.0, 0.5):
            rows = [
                row
                for row in records
                if row["panel"] == panel and row["cell"]["regd_load_bias"] == bias
            ]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["family"]].append(row)
            strata = {}
            for family, selected in sorted(grouped.items()):
                cases = defaultdict(list)
                by_model = defaultdict(list)
                for row in selected:
                    cases[case_key(row["cell"])].append(row)
                    by_model[row["policy_id"]].append(row)
                qualified = all(row["qualified"] for row in selected)
                metrics = {}
                for metric in METRICS:
                    if not qualified:
                        metrics[metric] = {
                            "mean": None,
                            "case_sd": None,
                            "fixed_checkpoint_mean_sd": None,
                        }
                        continue
                    case_means = np.array(
                        [
                            np.mean([row["metrics"][metric] for row in values])
                            for values in cases.values()
                        ]
                    )
                    model_means = np.array(
                        [
                            np.mean([row["metrics"][metric] for row in values])
                            for values in by_model.values()
                        ]
                    )
                    metrics[metric] = {
                        "mean": float(case_means.mean()),
                        "case_sd": float(case_means.std(ddof=1)),
                        "fixed_checkpoint_mean_sd": (
                            float(model_means.std(ddof=1))
                            if len(model_means) > 1
                            else None
                        ),
                        "fixed_checkpoint_mean_min": float(model_means.min()),
                        "fixed_checkpoint_mean_max": float(model_means.max()),
                    }
                strata[family] = {
                    "case_count": len(cases),
                    "fixed_policy_count": len(by_model),
                    "episode_count": len(selected),
                    "complete_count": sum(row["complete"] for row in selected),
                    "qualified_count": sum(row["qualified"] for row in selected),
                    "all_qualified": qualified,
                    "metrics": metrics,
                    "minimum_service_ratio": (
                        min(
                            row["metrics"]["delivered_service_ratio"]
                            for row in selected
                        )
                        if all(row["complete"] for row in selected)
                        else None
                    ),
                }
            output[panel][str(bias)] = {
                "families": strata,
                "observed_relief_cash_frontier": point_frontier(strata),
                "frontier_is_descriptive_not_confirmed_nondominance": True,
            }
    return output


def contrast(
    protocol, records, panel, bias, candidate_ids, comparator_id, indices, dates, seeds
):
    by_key = {
        (row["policy_id"], row["cell"]["start_day"], row["cell"]["seed"]): row
        for row in records
        if row["panel"] == panel and row["cell"]["regd_load_bias"] == bias
    }
    paired = []
    for day in dates:
        row = []
        for seed in seeds[day]:
            candidate = [
                by_key[(identifier, day, seed)] for identifier in candidate_ids
            ]
            comparator = by_key[(comparator_id, day, seed)]
            if not all(item["qualified"] for item in [*candidate, comparator]):
                return {
                    "available": False,
                    "reason": "ineligible_incomplete_or_disqualified_cohort",
                }
            row.append((candidate, comparator))
        paired.append(row)
    output, replicates = {}, {}
    for metric in protocol["analysis"]["primary_metrics"]:
        differences = np.array(
            [
                [
                    np.mean([item["metrics"][metric] for item in candidate])
                    - comparator["metrics"][metric]
                    for candidate, comparator in row
                ]
                for row in paired
            ]
        )
        output[metric], replicates[metric] = paired_estimate(differences, indices)
    thresholds = protocol["analysis"]["point_thresholds"]
    event = np.ones(indices.shape[0], dtype=bool)
    for metric, threshold in thresholds.items():
        event &= replicates[metric] > threshold
    return {
        "available": True,
        "metrics": output,
        "bootstrap_fraction_above_both_thresholds": float(event.mean()),
        "not_a_p_value_or_posterior_probability": True,
        "dates": dates,
        "fixed_candidate_ids": candidate_ids,
        "comparator_id": comparator_id,
        "bias": bias,
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "date_robust_point_label": all(
            all(
                value > thresholds[metric] for value in info["leave_one_date_out_means"]
            )
            for metric, info in output.items()
        ),
    }


def confirmation_inference(protocol, records):
    candidate_ids = [
        row["id"]
        for row in protocol["policies"]
        if row["family"] == protocol["candidate_family"]
    ]
    if len(candidate_ids) != 4:
        raise ValueError("Keep all four fixed candidate checkpoints")
    comparator_id = protocol["primary_comparator_id"]
    contrasts = {}
    for panel, tag in (("calendar", 11), ("peak_noise", 22)):
        dates, seeds, indices = noise_indices(
            protocol["panels"][panel]["cells"],
            protocol["analysis"]["bootstrap_replicates"],
            protocol["analysis"]["bootstrap_seed"],
            tag,
        )
        contrasts[panel] = {}
        for bias in (0.0, 0.5):
            result = contrast(
                protocol,
                records,
                panel,
                bias,
                candidate_ids,
                comparator_id,
                indices,
                dates,
                seeds,
            )
            result["per_fixed_training_seed"] = {
                identifier: contrast(
                    protocol,
                    records,
                    panel,
                    bias,
                    [identifier],
                    comparator_id,
                    indices,
                    dates,
                    seeds,
                )
                for identifier in candidate_ids
            }
            contrasts[panel][str(bias)] = result
    physical = [
        row
        for row in records
        if row["panel"] in protocol["analysis"]["physical_gate_panels"]
        and row["policy_id"] in [*candidate_ids, comparator_id]
    ]
    expected = 5 * sum(
        len(protocol["panels"][panel]["cells"])
        for panel in protocol["analysis"]["physical_gate_panels"]
    )
    if len(physical) != expected:
        raise ValueError("Primary gate cohort is incomplete")
    all_qualified = all(row["qualified"] for row in physical)
    primary = contrasts["calendar"]["0.5"]
    zero = contrasts["calendar"]["0.0"]
    numeric_success = (
        primary["available"]
        and zero["available"]
        and all(
            primary["metrics"][metric]["mean"] > threshold
            and primary["metrics"][metric]["lower_025"] > threshold
            for metric, threshold in protocol["analysis"]["point_thresholds"].items()
        )
        and zero["metrics"]["cashflow_26h_usd"]["mean"] >= 0
    )
    if all_qualified and numeric_success:
        status = "supported_on_the_prespecified_conditional_panel"
    elif not all_qualified:
        status = "not_admitted_physical_or_service_gate_failure"
    elif primary["available"] and any(
        primary["metrics"][metric]["mean"] <= threshold
        for metric, threshold in protocol["analysis"]["point_thresholds"].items()
    ):
        status = "joint_improvement_not_supported_by_point_estimates"
    else:
        status = "joint_improvement_not_established_by_bounds_or_zero_bias_guard"
    return {
        "status": status,
        "primary_comparison": primary,
        "all_primary_physical_slots": expected,
        "qualified_primary_physical_slots": sum(row["qualified"] for row in physical),
        "all_primary_physical_slots_qualified": all_qualified,
        "contrasts": contrasts,
        "bounds_condition_on_fixed_dates_and_fixed_checkpoints": True,
        "not_calendar_or_training_seed_population_bounds": True,
        "old_failed_promotion_gate_still_failed": True,
    }


def zero_parity(protocol, records):
    by_key = {
        (row["panel"], case_key(row["cell"])): row
        for row in records
        if row["policy_id"] == "script_zero"
    }
    checks = 0
    for panel, references in protocol["zero_reference_evidence"].items():
        for reference in references:
            if file_hash(reference["path"]) != reference["sha256"]:
                raise ValueError("Frozen analytic reference changed")
            old = read_json(reference["path"])
            new = by_key[(panel, case_key(reference["cell"]))]
            if bool(old["qualified"]) != new["qualified"]:
                raise AssertionError("Zero-overlay qualification changed")
            for metric, value in episode_metrics(old).items():
                close(new["metrics"][metric], value, tolerance=1e-6)
                checks += 1
    return checks


def main(stage):
    cap_cpu(32)
    protocol = read_json(ROOT / "registration.json")
    verify(protocol)
    panels = (
        ["development", "regression"]
        if stage == "development"
        else ["calendar", "peak_noise"]
    )
    if stage == "final":
        opening = read_json(ROOT / "final_opening.json")
        if opening["registration_id"] != protocol["registration_id"]:
            raise ValueError("Final opening differs from the frozen study")
    records = collect_panels(protocol, panels)
    summary = summarize(protocol, records, panels)
    report = {
        "registration_id": protocol["registration_id"],
        "stage": stage,
        "episode_count": len(records),
        "new_episode_count": sum(not row["reused"] for row in records),
        "qualified_count": sum(row["qualified"] for row in records),
        "raw_checks": sum(row["raw_checks"] for row in records),
        "raw_trace_hash_checks": sum(row["raw_file_hashes_checked"] for row in records),
        "paired_exogenous_and_stress_masks_verified": True,
        "source_hashes": protocol["source_hashes"],
        "summary": summary,
        "inference": confirmation_inference(protocol, records)
        if stage == "final"
        else None,
        "technical_validation_passed": True,
    }
    if stage == "development":
        report["zero_overlay_matches_frozen_analytic_checks"] = zero_parity(
            protocol, records
        )
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    cases_path = destination / f"{stage}_cases.json"
    report_path = destination / f"{stage}_report.json"
    if cases_path.exists() or report_path.exists():
        raise FileExistsError("Never overwrite an audited confirmation report")
    write_json(cases_path, records)
    report["case_records_sha256"] = file_hash(cases_path)
    write_json(report_path, report)
    csv_rows = []
    for panel, strata in summary.items():
        for bias, data in strata.items():
            for family, info in data["families"].items():
                csv_rows.append(
                    {
                        "panel": panel,
                        "bias": bias,
                        "family": family,
                        "qualified": info["qualified_count"],
                        "episodes": info["episode_count"],
                        "minimum_service_ratio": info["minimum_service_ratio"],
                        **{
                            metric: values["mean"]
                            for metric, values in info["metrics"].items()
                        },
                    }
                )
    pd.DataFrame(csv_rows).to_csv(destination / f"{stage}_summary.csv", index=False)
    verify(protocol)
    print(
        json.dumps(
            {
                "episodes": report["episode_count"],
                "qualified": report["qualified_count"],
                "raw_checks": report["raw_checks"],
                "status": report["inference"]["status"]
                if stage == "final"
                else "technical_validation_passed",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("development", "final"))
    main(parser.parse_args().stage)
