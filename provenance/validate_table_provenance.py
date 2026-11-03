#!/usr/bin/env python3
"""Read-only, standard-library validation of the released PID table evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

TABLE_COUNTS = {
    "T2a": 9,
    "T2b": 9,
    "Q1": 9,
    "Q2": 6,
    "Q3": 3,
    "Q4": 9,
    "Q5": 5,
    "Q6": 18,
    "Q7": 10,
    "Q8": 10,
    "Q9": 11,
    "Q10": 11,
    "Q11": 8,
}
VALUE_COUNTS = {
    "rule_original_grid": 54,
    "rule_original_quality": 45,
    "final_b_rule": 45,
    "historical_economic": 90,
    "latest_calendar_0.5": 70,
    "latest_calendar_0.0": 70,
    "latest_peak_noise_0.5": 77,
    "latest_peak_noise_0.0": 77,
}
APPROVED_INPUTS = frozenset(
    {
        "comparison_values",
        "reproduction_values",
        "tex",
        "low_tracking",
        "earlier_economic",
        "economic_original_peak",
        "confirmation",
        "pid_report",
        "pid_audit_supplement",
        "pid_reproduction",
        "low_case_records",
        "confirmation_development_cases",
        "confirmation_final_cases",
        "pid_cases",
    }
)
ALIASES = {
    "tracking_rmse_kw": "rmse_kw",
    "cash_vs_pid_usd": "cash_delta_vs_pid_usd",
}
REPRODUCTION_LABELS = {
    "stress_observed_regd": "Observed RegD (dimensionless)",
    "stress_actual_commitment_mw": "Actual commitment (MW)",
    "stress_demanded_response_mw": "Requested response (MW)",
    "stress_nominal_facility_mw": "Nominal-reference facility power (MW)",
    "stress_actual_facility_mw": "PID facility power (MW)",
    "stress_hvac_effort": "HVAC effort (0--1)",
    "stress_cooling_relief_mw": "Cooling-related relief (MW)",
    "stress_other_facility_relief_mw": "Remaining facility relief (MW)",
    "reported_formula_relief_mw": "Total facility relief (MW)",
}
ARCHIVE_NUMERIC_LABELS = {
    "Q1.facility_relief": "Facility relief (MW)",
    "Q1.LFR": "LFR (%)",
    "Q1.SPDR": "SPDR (%)",
    "Q1.LWSR": "LWSR (%)",
    "Q1.LWCF": "LW-CF (%)",
    "Q1.regulation_revenue_change": "Regulation revenue change (%)",
    "Q1.throughput_reward_change": "Throughput reward change (per tick)",
    "Q1.SOC_reward_change": "SOC reward change (per tick)",
    "Q1.thermal_reward_change": "Thermal reward change (per tick)",
}
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
SHA256 = re.compile(r"[0-9a-f]{64}")


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def finite(value, label):
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{label}: expected a finite number",
    )
    return value


def near(actual, expected, label, *, rel_tol=1e-10, abs_tol=1e-8):
    finite(actual, label)
    finite(expected, label)
    require(
        math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{label}: numeric mismatch ({actual!r} != {expected!r})",
    )


def relative_locator(value):
    require(isinstance(value, str) and value, "Empty or non-string locator")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and "\\" not in value
        and ":" not in value
        and all(part not in ("", ".", "..") for part in value.split("/")),
        "Public input locators must be normalized repository-relative POSIX paths",
    )
    return value


def contained_path(root, value):
    relative_locator(value)
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise ValidationError(
            "Input locator escapes the declared repository root"
        ) from None
    return candidate


def decode_json(raw, label):
    def reject_constant(_):
        raise ValidationError(f"{label}: non-finite JSON constant")

    try:
        return json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError(f"{label}: invalid JSON ({type(exc).__name__})") from None


def json_pointer(document, pointer):
    require(pointer == "" or pointer.startswith("/"), "Invalid JSON pointer")
    current = document
    for token in pointer.split("/")[1:] if pointer else []:
        token = token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                require(token.isdigit(), "Non-numeric array pointer")
                current = current[int(token)]
            else:
                current = current[token]
        except (KeyError, IndexError, TypeError):
            raise ValidationError(f"Unresolved JSON pointer: {pointer}") from None
    return current


def cell_key(row):
    cell = row["cell"]
    return cell["start_day"], cell["seed"], float(cell["regd_load_bias"])


def paired_key(row):
    return (row["panel"],) + cell_key(row)


def normalize_label(text):
    text = text.replace("$", "")
    text = re.sub(r"\^\{?\\dagger\}?", "", text)
    text = re.sub(r"\\(?:textbf|mathbf|textit|mathrm)\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\Phi", "Phi").replace(r"\%", "%")
    return re.sub(r"\s+", "", text).replace("{", "").replace("}", "")


def tabular_rows(tex, label_columns):
    rows = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%") or "&" not in line:
            continue
        columns = line.split("&")
        if len(columns) > label_columns and NUMBER.search(columns[label_columns]):
            rows.append(columns)
    return rows


def mean_sd(values):
    require(bool(values), "Cannot aggregate an empty cohort")
    for value in values:
        finite(value, "Aggregation input")
    return statistics.mean(values), statistics.stdev(values) if len(
        values
    ) > 1 else None


def checkpoint_averaged_values(records, metric, *, nested=False):
    groups = defaultdict(list)
    for row in records:
        value = row["metrics"].get(metric) if nested else row.get(metric)
        groups[cell_key(row)[:2]].append(value)
    return [
        statistics.mean(values)
        for values in groups.values()
        if all(value is not None for value in values)
    ]


class Sources:
    def __init__(self, root, mapping, overrides=None, check_hashes=False):
        self.root = Path(root).resolve()
        self.mapping = mapping
        self.overrides = overrides or {}
        self.check_hashes = check_hashes
        self.cache = {}
        self.receipts = {}
        require(
            set(self.overrides) <= APPROVED_INPUTS,
            "Input-map contains a source outside the approved numeric/TeX allowlist",
        )
        for locator in self.overrides.values():
            relative_locator(locator)

    def read(self, alias):
        require(alias in APPROVED_INPUTS, f"Disallowed validator input alias: {alias}")
        if alias == "tex":
            require(
                "tex" in self.overrides,
                "Approved renderer-generated TeX must be supplied explicitly; "
                "no original-TeX fallback",
            )
        if alias in self.cache:
            return self.cache[alias]
        entry = self.mapping["sources"][alias]
        locator = self.overrides.get(alias, entry["path"])
        relative_locator(locator)
        require(
            PurePosixPath(locator).suffix == (".tex" if alias == "tex" else ".json"),
            f"{alias}: only approved JSON or TeX input is accepted",
        )
        path = contained_path(self.root, locator)
        try:
            raw = path.read_bytes()
        except OSError:
            raise ValidationError(
                f"Missing or unreadable approved input: {locator}. "
                "Restore the local released scalar evidence; "
                "public artifact URLs may still be pending."
            ) from None
        if alias == "tex":
            try:
                self.set_generated_tex(locator, raw.decode("utf-8"))
            except UnicodeDecodeError:
                raise ValidationError("tex: invalid UTF-8") from None
            return self.cache[alias]
        observed = hashlib.sha256(raw).hexdigest()
        recorded = entry.get("sha256")
        if "checksum_ref" in entry:
            recorded = self.ref(entry["checksum_ref"])
        if recorded is not None:
            require(
                bool(SHA256.fullmatch(recorded)), f"{alias}: invalid recorded SHA256"
            )
        if self.check_hashes and recorded is not None:
            require(
                observed == recorded, f"{alias}: original-source byte SHA256 mismatch"
            )
        self.receipts[alias] = {
            "alias": alias,
            "path": locator,
            "observed_sha256": observed,
            "recorded_source_sha256": recorded,
            "byte_identical_to_recorded": observed == recorded if recorded else None,
        }
        try:
            self.cache[alias] = (
                raw.decode("utf-8") if alias == "tex" else decode_json(raw, alias)
            )
        except UnicodeDecodeError:
            raise ValidationError(f"{alias}: invalid UTF-8") from None
        return self.cache[alias]

    def set_generated_tex(self, locator, text):
        relative_locator(locator)
        require(
            PurePosixPath(locator).suffix == ".tex" and isinstance(text, str),
            "Generated numeric TeX needs a relative .tex locator and text",
        )
        self.overrides["tex"] = locator
        self.cache["tex"] = text
        self.receipts["tex"] = {
            "alias": "tex",
            "path": locator,
            "observed_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "recorded_source_sha256": None,
            "historical_original_tex_sha256": self.mapping["sources"]["tex"]["sha256"],
            "byte_identical_to_recorded": None,
            "role": (
                "generated_table_view_validated_numerically_"
                "not_by_original_layout_bytes"
            ),
        }

    def ref(self, reference):
        require("#" in reference, "Source reference lacks a JSON pointer separator")
        alias, pointer = reference.split("#", 1)
        return json_pointer(self.read(alias), pointer)


class Validator:
    def __init__(self, mapping, contracts, acceptance, sources):
        self.mapping = mapping
        self.contracts = contracts
        self.acceptance = acceptance
        self.sources = sources
        self.counts = Counter()
        self.expected_tex = {}
        self.seen_values = set()

    def metadata(self):
        tables = self.mapping["tables"]
        require(
            [(table["id"], len(table["rows"])) for table in tables]
            == list(TABLE_COUNTS.items()),
            "Table order/row coverage differs from the frozen 118-row inventory",
        )
        row_ids = [row["id"] for table in tables for row in table["rows"]]
        require(
            len(row_ids) == len(set(row_ids)) == 118, "Duplicate or missing row IDs"
        )
        require(
            self.acceptance["verification_snapshot"]["row_counts"] == TABLE_COUNTS,
            "Acceptance row counts disagree with the mapping",
        )
        head = self.mapping["source_repository_head"]
        require(
            self.contracts["source_repository_head"]
            == head
            == self.acceptance["source_repository_head"],
            "Source snapshot identities disagree",
        )
        for source in self.mapping["sources"].values():
            relative_locator(source["path"])
        models = self.contracts["model_catalog"]
        require(
            len(models) == len({model["id"] for model in models}) == 41,
            "Expected 41 unique declared checkpoint identities",
        )
        self.models = {model["id"]: model for model in models}
        for model in models:
            relative_locator(model["path"])
            require(
                bool(SHA256.fullmatch(model["sha256"])), "Invalid checkpoint digest"
            )
        for family in self.contracts["controller_families"].values():
            require(
                set(family["model_ids"]) <= self.models.keys(), "Unresolved model ID"
            )
        for name in ("corrected_selected_sac", "bc", "guided_sac", "pid_reproduction"):
            protocol = self.contracts["training_protocols"][name]
            require(
                "execution_commit" in protocol and protocol["execution_commit"] is None,
                f"{name}: preserve explicit null execution_commit",
            )
        archive = next(table for table in tables if table["id"] == "Q1")
        require(
            archive["default_cell_evidence_status"] == "reported_only",
            "Archival cells must retain reported-only provenance",
        )
        exceptions = [row for row in archive["rows"] if "cell_evidence_override" in row]
        require(
            len(exceptions) == 1
            and exceptions[0]["id"] == "Q1.facility_relief"
            and exceptions[0]["cell_evidence_override"]["column_index"] == 1
            and exceptions[0]["cell_evidence_override"]["family"] == "legacy_pid",
            "Only the legacy PID facility-relief mean/SD have a reproduction exception",
        )
        for table in tables:
            for row in table["rows"]:
                family = row.get("family", table.get("family"))
                if family:
                    require(
                        family in self.contracts["controller_families"],
                        f"{row['id']}: unknown controller family",
                    )
        self.counts["declared_checkpoint_identities"] = 41

    def source_row(self, table, row):
        source = self.sources.ref(row["source_ref"])
        adapter = row.get("source_adapter", table.get("source_adapter"))
        if adapter == "low":
            require(
                source["family"] == row["family"], f"{row['id']}: wrong source family"
            )
            require(
                source["panel"] == table["source_panel"], f"{row['id']}: wrong panel"
            )
            require(source["macro"] == table["macro"], f"{row['id']}: wrong macro")
            passed = [source["qualified_policy_cases"], source["planned_policy_cases"]]
            complete = source["complete_cohort"]
        elif adapter == "earlier_economic":
            gate = self.sources.ref(row["gate_ref"])
            passed = [gate["qualified_policy_cases"], gate["policy_case_count"]]
            complete = gate["complete_cohort"]
            require(
                gate["all_qualified"] == (passed[0] == passed[1]),
                f"{row['id']}: inconsistent gate flag",
            )
            require(
                source["case_count"] == 15, f"{row['id']}: wrong original-peak count"
            )
            model_count = (
                len(self.contracts["controller_families"][row["family"]]["model_ids"])
                or 1
            )
            require(
                gate["model_count"] == model_count and passed[1] == 30 * model_count,
                f"{row['id']}: wrong policy-case denominator",
            )
            require(
                gate["minimum_service_ratio"] >= 0.9,
                f"{row['id']}: service-floor failure",
            )
            near(
                gate["biased_cash_delta_vs_original_matched_pid"],
                source["metrics"]["cashflow_26h_usd"]["mean"]
                - self.sources.ref(table["cash_reference_ref"])["metrics"][
                    "cashflow_26h_usd"
                ]["mean"],
                f"{row['id']}: source cash contrast",
            )
            self.counts["earlier_economic_report_bound_rows"] += 1
        elif adapter == "confirmation":
            passed = [source["qualified_count"], source["episode_count"]]
            complete = source["complete_count"] == source["episode_count"]
            if "zero_bias_source_ref" in row:
                zero = self.sources.ref(row["zero_bias_source_ref"])
                passed[0] += zero["qualified_count"]
                passed[1] += zero["episode_count"]
                complete = complete and zero["complete_count"] == zero["episode_count"]
        elif adapter == "pid":
            passed = [source["qualified"], source["episodes"]]
            complete = source["all_complete"]
        else:
            raise ValidationError(f"{row['id']}: unknown source adapter")
        require(complete, f"{row['id']}: incomplete cohort")
        require(passed == row["pass"], f"{row['id']}: incorrect pass/planned counts")
        diagnostic = complete and passed[0] != passed[1]
        require(
            diagnostic == row.get("diagnostic", False), f"{row['id']}: diagnostic flag"
        )
        self.counts["gate_rows"] += 1
        return source, adapter, diagnostic

    def main_tables(self, *, table_ids=None):
        values = self.sources.read("comparison_values")
        require(
            Counter(value["scope"] for value in values) == VALUE_COUNTS,
            "Main values manifest scope/count mismatch",
        )
        index = {
            (value["scope"], value["family"], value["metric"]): value
            for value in values
        }
        require(
            len(index) == len(values) == 528, "Duplicate or missing comparison values"
        )
        for table in self.mapping["tables"]:
            if table_ids is not None and table["id"] not in table_ids:
                continue
            if "value_scope" not in table:
                continue
            specs = self.mapping["metric_sets"][table["metric_set"]]
            displayed = defaultdict(list)
            for row in table["rows"]:
                source, adapter, diagnostic = self.source_row(table, row)
                expected = []
                for spec in specs:
                    metric = spec["metric"]
                    key = (table["value_scope"], row["family"], metric)
                    require(
                        key in index, f"{row['id']}: missing displayed metric {metric}"
                    )
                    value = index[key]
                    self.seen_values.add(key)
                    source_metric = (
                        ALIASES.get(metric, metric) if adapter == "low" else metric
                    )
                    if metric == "cash_vs_pid_usd" and adapter != "low":
                        baseline = self.sources.ref(table["cash_reference_ref"])
                        mean = (
                            source["metrics"]["cashflow_26h_usd"]["mean"]
                            - baseline["metrics"]["cashflow_26h_usd"]["mean"]
                        )
                    else:
                        mean = source["metrics"][source_metric]["mean"]
                    near(
                        value["value"],
                        mean,
                        f"{row['id']}/{metric}",
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                    factor, digits = spec.get("factor", 1), spec["digits"]
                    require(
                        value["factor"] == factor and value["digits"] == digits,
                        f"{row['id']}/{metric}: wrong display scale/precision",
                    )
                    token = f"{mean * factor:.{digits}f}"
                    require(
                        value["display"] == token,
                        f"{row['id']}/{metric}: wrong display",
                    )
                    require(
                        value["diagnostic"] == diagnostic
                        and value["averaging_complete_cohort"] is True,
                        f"{row['id']}/{metric}: cohort flags",
                    )
                    tokens = [token]
                    if table["show_sd"]:
                        sd = source["metrics"][source_metric]["sd"]
                        near(value["sd_value"], sd, f"{row['id']}/{metric}: SD")
                        sd_token = f"{sd * factor:.{digits}f}"
                        require(
                            value["sd_display"] == sd_token,
                            f"{row['id']}/{metric}: SD display",
                        )
                        tokens.append(sd_token)
                        self.counts["SD_cells"] += 1
                    expected.append(tokens)
                    displayed[metric].append((float(token), diagnostic, value["bold"]))
                    self.counts["main_value_cells"] += 1
                expected.append([str(count) for count in row["pass"]])
                self.expected_tex[row["id"]] = expected
            for metric, entries in displayed.items():
                qualified = [
                    value for value, diagnostic, _ in entries if not diagnostic
                ]
                require(
                    bool(qualified),
                    f"{table['id']}/{metric}: no eligible displayed value",
                )
                optimum = (
                    min
                    if metric in {"tracking_rmse_kw", "battery_throughput_26h_mwh"}
                    else max
                )(qualified)
                for value, diagnostic, bold in entries:
                    require(
                        bold == (not diagnostic and value == optimum),
                        f"{table['id']}/{metric}: qualified printed-precision "
                        "emphasis mismatch",
                    )
        expected_count = sum(
            len(table["rows"]) * len(self.mapping["metric_sets"][table["metric_set"]])
            for table in self.mapping["tables"]
            if "value_scope" in table
            and (table_ids is None or table["id"] in table_ids)
        )
        require(
            len(self.seen_values) == expected_count,
            "Not all selected main comparison values were consumed",
        )

    def other_tables(self):
        reproduction_values = self.sources.read("reproduction_values")
        require(
            len(reproduction_values) == 9,
            "Expected nine PID reproduction metric records",
        )
        reproduction = self.sources.read("pid_reproduction")
        for table in self.mapping["tables"]:
            for index, row in enumerate(table["rows"]):
                if table["id"] == "Q1":
                    require(
                        len(row["means"]) == len(row["sample_sds"]) == 4,
                        "Archive must retain four reported controller columns",
                    )
                    digits = [2, 2, 3, 3, 2, 2, 3, 3, 3][index]
                    self.expected_tex[row["id"]] = [
                        [f"{mean:.{digits}f}", f"{sd:.{digits}f}"]
                        for mean, sd in zip(row["means"], row["sample_sds"])
                    ]
                    self.counts["archival_reported_pairs"] += 4
                    if index == 0:
                        old = reproduction["summary"]["legacy_v0"]
                        require(
                            f"{old['reported_formula_relief_mw']:.2f}"
                            == f"{row['means'][1]:.2f}"
                            and f"{old['relief_case_sd_mw']:.2f}"
                            == f"{row['sample_sds'][1]:.2f}",
                            "Narrow archival PID relief reproduction does not agree",
                        )
                elif table["id"] in {"Q2", "Q3"}:
                    metric = row["metric"]
                    old = reproduction["summary"]["legacy_v0"][metric]
                    new = reproduction["summary"]["physical_v1"][metric]
                    values = self.sources.ref(row["values_ref"])
                    require(
                        values["metric"] == metric,
                        f"{row['id']}: wrong reproduction metric",
                    )
                    for field, expected in (
                        ("earlier", old),
                        ("corrected", new),
                        ("change", new - old),
                    ):
                        near(values[field], expected, f"{row['id']}/{field}")
                    display = [f"{old:.4f}", f"{new:.4f}", f"{new - old:+.4f}"]
                    require(
                        display == row["display"] == values["displayed"],
                        f"{row['id']}: reproduction display mismatch",
                    )
                    self.expected_tex[row["id"]] = [[token] for token in display]
                    self.counts["reproduction_numeric_cells"] += 3
                elif table["id"] == "Q5":
                    rule = self.sources.ref(row["rule_source_ref"])
                    full = self.sources.ref(row["full_residual_source_ref"])
                    for source, macro, counts in (
                        (rule, "rule", row["rule_pass"]),
                        (full, "full_residual", row["full_residual_pass"]),
                    ):
                        require(
                            source["family"] == row["family"]
                            and source["macro"] == macro
                            and source["panel"] == "regression",
                            f"{row['id']}: macro pairing",
                        )
                        require(
                            [
                                source["qualified_policy_cases"],
                                source["planned_policy_cases"],
                            ]
                            == counts,
                            f"{row['id']}: source macro counts",
                        )
                    a, b = rule["metrics"], full["metrics"]
                    numbers = [
                        a["stress_facility_relief_mw"]["mean"],
                        b["stress_facility_relief_mw"]["mean"],
                        b["cashflow_26h_usd"]["mean"] - a["cashflow_26h_usd"]["mean"],
                        a["rmse_kw"]["mean"],
                        b["rmse_kw"]["mean"],
                    ]
                    display = [
                        f"{number:.{column['digits']}f}"
                        for number, column in zip(numbers, table["columns"])
                    ]
                    require(
                        display == row["display"], f"{row['id']}: macro-only values"
                    )
                    self.expected_tex[row["id"]] = [[token] for token in display]
                    self.counts["macro_only_numeric_cells"] += 5
                elif table["id"] == "Q11":
                    source = self.sources.ref(row["source_ref"])
                    specs = self.mapping["metric_sets"]["resources"]
                    display = [
                        format(
                            source[spec["metric"]] * spec.get("factor", 1),
                            f".{spec['digits']}f",
                        )
                        for spec in specs
                    ]
                    require(display == row["display"], f"{row['id']}: resource display")
                    require(
                        [source["qualified"], source["episodes"]] == row["pass"],
                        f"{row['id']}: resource pass counts",
                    )
                    self.expected_tex[row["id"]] = [[token] for token in display] + [
                        [str(count) for count in row["pass"]]
                    ]
                    self.counts["resource_numeric_cells"] += 5

    def tex(self):
        text = self.sources.read("tex")
        blocks = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, re.S)
        require(len(blocks) == 13, "Expected thirteen generated TeX tabular parts")
        families = self.contracts["controller_families"]
        values = {
            (value["scope"], value["family"], value["metric"]): value
            for value in self.sources.read("comparison_values")
        }
        for table, block in zip(self.mapping["tables"], blocks):
            start = 2 if table["id"] == "Q11" else 1
            rows = tabular_rows(block, start)
            require(len(rows) == len(table["rows"]), f"{table['id']}: TeX row count")
            for row, columns in zip(table["rows"], rows):
                actual = [NUMBER.findall(column) for column in columns[start:]]
                require(
                    actual == self.expected_tex[row["id"]],
                    f"{row['id']}: printed numeric entries/precision differ",
                )
                if table["id"] == "Q1":
                    label = row["source_key"]
                elif table["id"] in {"Q2", "Q3"}:
                    label = REPRODUCTION_LABELS[row["metric"]]
                elif table["id"] == "Q11":
                    panel = (
                        "New calendar" if row["panel"] == "calendar" else "Known peak"
                    )
                    label = f"{panel}/{row['bias']}"
                    variant = "Native" if row["family"] == "pid_native" else "Full work"
                    require(
                        normalize_label(columns[1]) == normalize_label(variant),
                        f"{row['id']}: PID variant label",
                    )
                else:
                    label = families[row["family"]]["label"]
                labels = {normalize_label(label)}
                if table["id"] == "Q1":
                    labels.add(normalize_label(ARCHIVE_NUMERIC_LABELS[row["id"]]))
                require(
                    normalize_label(columns[0]) in labels,
                    f"{row['id']}: controller/metric label differs",
                )
                if "value_scope" in table:
                    require(
                        (r"\dagger" in columns[0]) == row.get("diagnostic", False),
                        f"{row['id']}: missing or spurious diagnostic marker",
                    )
                    for column, spec in zip(
                        columns[1:], self.mapping["metric_sets"][table["metric_set"]]
                    ):
                        value = values[
                            (table["value_scope"], row["family"], spec["metric"])
                        ]
                        require(
                            (r"\mathbf" in column) == value["bold"],
                            f"{row['id']}: printed emphasis differs",
                        )
                self.counts["TeX_rows"] += 1
                self.counts["TeX_numeric_entries"] += sum(
                    len(tokens) for tokens in actual
                )
        require(
            self.counts["TeX_rows"] == 118
            and self.counts["TeX_numeric_entries"] == 1026,
            "Exhaustive TeX numeric coverage failed",
        )

    def low_aggregations(self, *, table_ids=None):
        records = self.sources.read("low_case_records")
        report = self.sources.read("low_tracking")
        groups = defaultdict(list)
        peers = {}
        for row in records:
            key = (row["panel"], row["macro"], row["family"])
            groups[key].append(row)
            peer_key = (row["panel"], row["macro"], row["policy_id"]) + cell_key(row)
            require(peer_key not in peers, "Duplicate low-level policy/cell record")
            peers[peer_key] = row
        tables = [
            table
            for table in self.mapping["tables"]
            if table_ids is None or table["id"] in table_ids
        ]
        indices = set()
        for table in tables:
            for row in table["rows"]:
                for name in (
                    "source_ref",
                    "rule_source_ref",
                    "full_residual_source_ref",
                ):
                    reference = row.get(name, "")
                    if reference.startswith("low_tracking#/aggregates/"):
                        indices.add(int(reference.rsplit("/", 1)[1]))
        for index in sorted(indices):
            aggregate = report["aggregates"][index]
            panel, macro, family = (
                aggregate[key] for key in ("panel", "macro", "family")
            )
            cohort = groups[(panel, macro, family)]
            planned = aggregate["planned_policy_cases"]
            require(
                len(cohort) == planned
                and sum(row["qualified"] for row in cohort)
                == aggregate["qualified_policy_cases"],
                f"Low aggregate {index}: counts",
            )
            expected_cells = {
                cell_key({"cell": cell}) for cell in report["panels"][panel]
            }
            require(
                {cell_key(row) for row in cohort} == expected_cells,
                f"Low aggregate {index}: incorrect calendar/noise/bias cells",
            )
            multiplicity = Counter(cell_key(row) for row in cohort)
            require(
                set(multiplicity.values()) == {aggregate["planned_models"]},
                f"Low aggregate {index}: missing checkpoint cell",
            )
            for row in cohort:
                require(row["complete_26h"], f"Low aggregate {index}: incomplete case")
                require(
                    row["floor_enforced"] == (macro == "full_residual"),
                    f"Low aggregate {index}: service-floor metadata differs",
                )
                expected_gate = row["physical_gate_qualified"] and (
                    row["delivered_service_gate"] if row["floor_enforced"] else True
                )
                require(
                    row["qualified"] == expected_gate,
                    f"Low aggregate {index}: qualification",
                )
                require(
                    (row["start_day"], row["env_seed"], float(row["bias"]))
                    == cell_key(row),
                    f"Low aggregate {index}: contradictory cell fields",
                )
                pid_key = (panel, macro, f"{macro}__pid") + cell_key(row)
                require(
                    pid_key in peers, "Missing matched historical PID scalar record"
                )
                near(
                    row["cash_delta_vs_pid_usd"],
                    row["cashflow_26h_usd"] - peers[pid_key]["cashflow_26h_usd"],
                    "Paired historical cash difference",
                )
                if row["bias"] == 0.5:
                    zero_key = (
                        panel,
                        macro,
                        row["policy_id"],
                        row["start_day"],
                        row["env_seed"],
                        0.0,
                    )
                    zero = peers[zero_key]["regulation_revenue_usd"]
                    near(
                        row["matched_bias_revenue_change_pct"],
                        100 * (row["regulation_revenue_usd"] / zero - 1),
                        "Own-controller paired bias revenue",
                    )
            biased = [row for row in cohort if row["bias"] == 0.5]
            for metric, saved in aggregate["metrics"].items():
                values = checkpoint_averaged_values(biased, metric)
                if saved["mean"] is None:
                    require(
                        not values,
                        f"Low aggregate {index}/{metric}: unexpected defined value",
                    )
                    continue
                mean, sd = mean_sd(values)
                near(mean, saved["mean"], f"Low aggregate {index}/{metric}: mean")
                near(sd, saved["sd"], f"Low aggregate {index}/{metric}: case SD")
                require(
                    len(values) == saved["defined_calendar_cases"],
                    f"Low aggregate {index}/{metric}: defined-case count",
                )
                self.counts["low_metric_mean_SD_pairs"] += 1
            self.counts["low_aggregate_rows"] += 1
        expected_rows = 23 if table_ids is None else len(indices)
        expected_metrics = (
            414 if table_ids is None
            else sum(len(report["aggregates"][index]["metrics"]) for index in indices)
        )
        require(
            self.counts["low_aggregate_rows"] == expected_rows
            and self.counts["low_metric_mean_SD_pairs"] == expected_metrics,
            "Low-level aggregation coverage differs from the selected scope",
        )
        for table in tables:
            if table["id"] != "Q5":
                continue
            for row in table["rows"]:
                paired = [
                    {
                        (record["training_seed"],) + cell_key(record)
                        for record in groups[("regression", macro, row["family"])]
                    }
                    for macro in ("rule", "full_residual")
                ]
                require(
                    paired[0] == paired[1], f"{row['id']}: unmatched macro contrast"
                )

    def confirmation_aggregations(self):
        for stage, source_alias, records_alias, total in (
            ("final", "confirmation", "confirmation_final_cases", 9460),
            (
                "development",
                "economic_original_peak",
                "confirmation_development_cases",
                2322,
            ),
        ):
            report = self.sources.read(source_alias)
            records = self.sources.read(records_alias)
            require(
                len(records) == total == report["episode_count"],
                f"{stage}: fixed cohort size changed",
            )
            groups = defaultdict(list)
            unique = set()
            for row in records:
                key = (
                    row["panel"],
                    str(float(row["cell"]["regd_load_bias"])),
                    row["family"],
                )
                groups[key].append(row)
                unique_key = paired_key(row) + (row["policy_id"],)
                require(unique_key not in unique, f"{stage}: duplicate policy/cell")
                unique.add(unique_key)
                require(
                    row["complete"] and row["qualified"],
                    f"{stage}: failed/incomplete fixed cohort",
                )
                if row["kind"] == "neural":
                    model_id = f"{row['family']}_{row['training_seed']}"
                    require(
                        model_id in self.models,
                        f"{stage}: unexpected neural checkpoint",
                    )
                    require(
                        row["checkpoint_sha256"] == self.models[model_id]["sha256"],
                        f"{stage}: checkpoint digest changed",
                    )
            require(
                len({row["policy_id"] for row in records}) == 43,
                f"{stage}: original 43-policy roster changed",
            )
            require(
                sum(bool(row["reused"]) for row in records)
                == (0 if stage == "final" else 864),
                f"{stage}: frozen neural reuse count differs",
            )
            require(
                report["qualified_count"] == total,
                f"{stage}: qualification total differs",
            )
            for panel, biases in report["summary"].items():
                for bias, summary in biases.items():
                    for family, aggregate in summary["families"].items():
                        cohort = groups.pop((panel, bias, family), [])
                        keys = {cell_key(row)[:2] for row in cohort}
                        policies = {row["policy_id"] for row in cohort}
                        count = len(cohort)
                        require(
                            count
                            == aggregate["episode_count"]
                            == aggregate["complete_count"]
                            == aggregate["qualified_count"],
                            f"{stage}/{panel}/{bias}/{family}: counts",
                        )
                        require(
                            len(keys) == aggregate["case_count"]
                            and len(policies) == aggregate["fixed_policy_count"]
                            and count == len(keys) * len(policies)
                            and aggregate["all_qualified"],
                            f"{stage}/{panel}/{bias}/{family}: averaging dimensions",
                        )
                        near(
                            min(
                                row["metrics"]["delivered_service_ratio"]
                                for row in cohort
                            ),
                            aggregate["minimum_service_ratio"],
                            "Minimum actual service ratio",
                        )
                        for metric, saved in aggregate["metrics"].items():
                            mean, sd = mean_sd(
                                checkpoint_averaged_values(cohort, metric, nested=True)
                            )
                            near(
                                mean,
                                saved["mean"],
                                f"{stage}/{panel}/{family}/{metric}: mean",
                            )
                            near(
                                sd,
                                saved["case_sd"],
                                f"{stage}/{panel}/{family}/{metric}: case SD",
                            )
                            per_policy = [
                                statistics.mean(
                                    row["metrics"][metric]
                                    for row in cohort
                                    if row["policy_id"] == policy
                                )
                                for policy in policies
                            ]
                            near(
                                min(per_policy),
                                saved["fixed_checkpoint_mean_min"],
                                "Fixed-checkpoint minimum",
                            )
                            near(
                                max(per_policy),
                                saved["fixed_checkpoint_mean_max"],
                                "Fixed-checkpoint maximum",
                            )
                            checkpoint_sd = (
                                statistics.stdev(per_policy)
                                if len(per_policy) > 1
                                else None
                            )
                            if checkpoint_sd is None:
                                require(
                                    saved["fixed_checkpoint_mean_sd"] is None,
                                    "Singleton policy must not acquire a checkpoint SD",
                                )
                            else:
                                near(
                                    checkpoint_sd,
                                    saved["fixed_checkpoint_mean_sd"],
                                    "Fixed-checkpoint SD",
                                )
                            self.counts["confirmation_metric_mean_caseSD_pairs"] += 1
            require(not groups, f"{stage}: unmatched family/cell groups")
        require(
            self.counts["confirmation_metric_mean_caseSD_pairs"] == 3968,
            "Confirmation/development aggregation coverage changed",
        )

    def pid_aggregations_and_pairing(self):
        records = self.sources.read("pid_cases")
        report = self.sources.read("pid_report")
        supplement = self.sources.read("pid_audit_supplement")
        final = self.sources.read("confirmation_final_cases")
        require(
            len(records) == report["episodes"] == report["complete"] == 440,
            "PID cohort/completion count changed",
        )
        require(
            sum(row["qualified"] for row in records) == report["qualified"] == 354,
            "PID qualification total changed",
        )
        inputs = defaultdict(set)
        for row in final:
            inputs[paired_key(row)].add(row["inputs_sha256"])
        require(
            len(inputs) == 220 and all(len(values) == 1 for values in inputs.values()),
            "Frozen confirmation lacks 220 consistently paired input digests",
        )
        groups = defaultdict(list)
        index = {}
        failures = defaultdict(set)
        for row in records:
            key = paired_key(row)
            require(
                row["inputs_sha256"] is not None
                and row["inputs_sha256"] in inputs[key],
                "PID/confirmation exogenous pairing differs",
            )
            require(row["complete"], "An incomplete PID entered a descriptive mean")
            require(
                row["metrics"]["delivered_service_ratio"] >= 0.9,
                "PID paid-service floor failure",
            )
            require(
                row["failure_reasons"]
                == ([] if row["qualified"] else ["terminal_soc"]),
                "PID failure reason is not the recorded service-end SOC reserve",
            )
            unique_key = key + (row["family"],)
            require(unique_key not in index, "Duplicate PID policy/cell")
            index[unique_key] = row
            groups[
                (row["panel"], str(float(row["cell"]["regd_load_bias"])), row["family"])
            ].append(row)
            if not row["qualified"]:
                failures[row["family"]].add(key)
        require(
            failures["pid_native"] == failures["pid_full_work"]
            and len(failures["pid_native"]) == 43,
            "PID failure sets differ",
        )
        for panel, biases in report["summary"].items():
            for bias, variants in biases.items():
                for family, summary in variants.items():
                    cohort = groups[(panel, bias, family)]
                    require(
                        len(cohort) == summary["episodes"] == summary["complete"],
                        "PID summary denominator differs",
                    )
                    require(
                        sum(row["qualified"] for row in cohort) == summary["qualified"],
                        "PID stratum qualification count differs",
                    )
                    require(
                        summary["all_complete"] and not summary["all_qualified"],
                        "PID diagnostic cohort incorrectly admitted",
                    )
                    for metric, saved in summary["metrics"].items():
                        mean, sd = mean_sd([row["metrics"][metric] for row in cohort])
                        near(mean, saved["mean"], "PID complete-case mean")
                        near(sd, saved["sd"], "PID complete-case SD")
                        require(
                            saved["admissible_mean"] is None,
                            "A failed PID cohort acquired an admissible mean",
                        )
                        self.counts["PID_metric_mean_SD_pairs"] += 1
        diagnostics = supplement["case_diagnostics"]
        require(len(diagnostics) == 440, "Missing PID resource cases")
        diagnostic_groups = defaultdict(list)
        diagnostic_keys = set()
        for row in diagnostics:
            key = paired_key(row) + (row["policy_id"],)
            require(
                key in index and key not in diagnostic_keys,
                "PID resource pairing mismatch",
            )
            diagnostic_keys.add(key)
            require(
                row["qualified"] == index[key]["qualified"],
                "PID resource gate mismatch",
            )
            require(
                abs(row["recovery_soc_error"]) <= 1e-6, "PID recovery SOC did not close"
            )
            if row["policy_id"] == "pid_full_work":
                require(
                    row["stress_mean_throttle"] == row["minimum_throttle"] == 1.0,
                    "Full-work PID throttle override changed",
                )
                require(
                    row["maximum_backlog_kwh"]
                    == row["positive_service_it_curtailment_mwh"]
                    == 0.0,
                    "Full-work PID work diagnostics changed",
                )
            diagnostic_groups[
                (
                    row["panel"],
                    str(float(row["cell"]["regd_load_bias"])),
                    row["policy_id"],
                )
            ].append(row)
        require(diagnostic_keys == set(index), "Missing PID resource counterpart")
        for panel, biases in supplement["summary"].items():
            for bias, variants in biases.items():
                for family, summary in variants.items():
                    cohort = diagnostic_groups[(panel, bias, family)]
                    for field in (
                        "stress_mean_throttle",
                        "stress_mean_it_curtailment_mw",
                        "positive_service_it_curtailment_mwh",
                    ):
                        near(
                            statistics.mean(row[field] for row in cohort),
                            summary[field],
                            "PID resource mean",
                        )
                    near(
                        max(row["maximum_backlog_kwh"] for row in cohort),
                        summary["worst_backlog_kwh"],
                        "Worst queue, not mean queue",
                    )
                    near(
                        min(row["terminal_service_soc"] for row in cohort),
                        summary["minimum_terminal_service_soc"],
                        "Minimum terminal SOC, not minimum intraepisode SOC",
                    )
                    require(
                        [sum(row["qualified"] for row in cohort), len(cohort)]
                        == [summary["qualified"], summary["episodes"]],
                        "PID resource counts",
                    )
                    self.counts["PID_resource_aggregate_rows"] += 1
        for row in records:
            if row["panel"] == "peak_noise" and row["family"] == "pid_native":
                other = index[paired_key(row) + ("pid_full_work",)]
                for metric in (
                    "stress_facility_relief_mw",
                    "stress_pcc_relief_mw",
                    "tracking_rmse_kw",
                    "cashflow_26h_usd",
                    "battery_throughput_26h_mwh",
                ):
                    require(
                        row["metrics"][metric] == other["metrics"][metric],
                        "Native/full-work peak-panel equality changed",
                    )
                self.counts["identical_peak_PID_pairs"] += 1
        require(
            self.counts["identical_peak_PID_pairs"] == 60,
            "Missing peak PID comparisons",
        )
        self.counts["paired_confirmation_cells"] = 220
        self.counts["PID_cases"] = 440

    def reproduction_aggregations(self):
        report = self.sources.read("pid_reproduction")
        cases = report["cases"]
        require(
            len(cases) == report["case_pair_count"] == 30
            and report["diagnostic_run_count"] == 60
            and report["per_mode_case_pairs"] == 15,
            "PID reproduction population changed",
        )
        require(
            report[
                "recovery_not_run_or_needed_for_this_service_only_metric_reproduction"
            ]
            and report["no_training_or_gain_changes"],
            "PID reproduction role changed",
        )
        expected = {(day, seed) for day in (207, 248, 249) for seed in range(100, 105)}
        metrics = list(REPRODUCTION_LABELS)
        for mode in ("legacy_v0", "physical_v1"):
            cohort = [row for row in cases if row["mode"] == mode]
            require(
                len(cohort) == 15
                and {(row["day"], row["seed"]) for row in cohort} == expected,
                "PID reproduction date/seed labels changed",
            )
            for row in cohort:
                require(
                    row["agent_ticks"] == row["baseline_ticks"] == 17280,
                    "Incomplete PID reproduction trajectory",
                )
                near(
                    row["stress_cooling_relief_mw"]
                    + row["stress_other_facility_relief_mw"],
                    row["reported_formula_relief_mw"],
                    "PID observed case power balance",
                )
            for metric in metrics:
                near(
                    statistics.mean(row[metric] for row in cohort),
                    report["summary"][mode][metric],
                    f"PID reproduction/{mode}/{metric}",
                )
                self.counts["reproduction_mode_metric_means"] += 1
            near(
                statistics.stdev(row["reported_formula_relief_mw"] for row in cohort),
                report["summary"][mode]["relief_case_sd_mw"],
                "PID reproduction sample SD",
            )

    def run(self):
        self.metadata()
        self.main_tables()
        self.other_tables()
        self.tex()
        self.low_aggregations()
        self.confirmation_aggregations()
        self.pid_aggregations_and_pairing()
        self.reproduction_aggregations()
        require(
            set(self.sources.cache) == APPROVED_INPUTS,
            "Unexpected input access or incomplete approved-input coverage",
        )
        self.counts["approved_input_files"] = len(self.sources.receipts)
        self.counts["recorded_source_byte_hashes_compared"] = sum(
            receipt["recorded_source_sha256"] is not None
            for receipt in self.sources.receipts.values()
        )
        self.counts["source_byte_hash_mismatches"] = sum(
            receipt["byte_identical_to_recorded"] is False
            for receipt in self.sources.receipts.values()
        )
        return {
            "schema": "dc2grid-portable-table-validation-v1",
            "status": "passed",
            "source_repository_head": self.mapping["source_repository_head"],
            "counts": dict(sorted(self.counts.items())),
            "inputs": [
                self.sources.receipts[key] for key in sorted(self.sources.receipts)
            ],
            "original_source_byte_hashes_enforced": self.sources.check_hashes,
            "limits": [
                (
                    "Q1 is frozen reported numeric evidence; only the narrow PID "
                    "relief pair has a separate recorded reproduction."
                ),
                (
                    "Four earlier absolute-economic Q6 rows are bound to approved "
                    "aggregate reports; archived episode/trace JSON is not "
                    "required or recomputed."
                ),
                (
                    "Checkpoint digests are declared identities here; checkpoint "
                    "archives and runtime implementations are not read."
                ),
                (
                    "The approved nine-page PDF is a frozen view; numeric table "
                    "TeX does not regenerate its layout or original document bytes."
                ),
                (
                    "Changed source-byte digests need a separately documented "
                    "release transformation; scientific-value validation does "
                    "not replace that receipt."
                ),
            ],
            "excluded_inputs_accessed": [],
            "public_artifact_url_status": "pending",
            "publication_authorization_decided_by_validator": False,
            "network_access": False,
            "input_files_modified": False,
        }


def metadata_document(directory, filename):
    try:
        raw = (directory / filename).read_bytes()
    except OSError:
        raise ValidationError(f"Missing metadata document: {filename}") from None
    return decode_json(raw, filename)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Validate all 118 PID table rows and 1,026 printed numeric entries "
            "without the original ZIP or private files."
        ),
        epilog=(
            "Read-only. Input paths are relative to --root. --tex may name "
            "approved renderer-generated TeX. No comparison_data blob, runtime "
            "code, checkpoints, PDF, trace CSV, bundles, or network access is needed."
        ),
    )
    parser.add_argument(
        "--root", required=True, type=Path, help="Root of the clean code/results export"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory holding the three provenance JSON documents",
    )
    parser.add_argument(
        "--tex",
        required=True,
        help=(
            "Required repository-relative locator of approved renderer-generated "
            "TeX; no original-file fallback"
        ),
    )
    parser.add_argument(
        "--input-map",
        help=(
            "Repository-relative JSON object mapping approved source aliases "
            "to relative locators"
        ),
    )
    parser.add_argument(
        "--check-source-hashes",
        action="store_true",
        help=(
            "Require original-byte SHA256 for numeric JSON inputs; generated "
            "TeX is checked numerically, not against original document bytes"
        ),
    )
    args = parser.parse_args(argv)
    try:
        mapping = metadata_document(args.metadata_dir, "table_row_provenance.json")
        contracts = metadata_document(
            args.metadata_dir, "controller_protocol_provenance.json"
        )
        acceptance = metadata_document(args.metadata_dir, "provenance_acceptance.json")
        overrides = {}
        if args.input_map:
            path = contained_path(args.root, args.input_map)
            try:
                overrides = decode_json(path.read_bytes(), "input-map")
            except OSError:
                raise ValidationError(
                    "Missing or unreadable repository-relative input-map"
                ) from None
            require(isinstance(overrides, dict), "Input-map must be a JSON object")
        if args.tex:
            overrides["tex"] = args.tex
        sources = Sources(args.root, mapping, overrides, args.check_source_hashes)
        result = Validator(mapping, contracts, acceptance, sources).run()
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        ZeroDivisionError,
        OverflowError,
    ) as exc:
        error = (
            str(exc)
            if isinstance(exc, ValidationError)
            else f"Malformed approved evidence structure ({type(exc).__name__})"
        )
        result = {
            "schema": "dc2grid-portable-table-validation-v1",
            "status": "failed",
            "error": error,
        }
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
