from prepare_table1 import original_table1
from table1_eval import MODEL_SPECS, PANEL_SPEC, ROW_SPECS, SERVICE_GATE_SPEC


def test_original_table1_is_nine_reported_rows_not_new_results():
    original = original_table1()
    assert len(original["rows"]) == 9
    assert original["reported_legacy_values_not_new_results"]
    assert not original["exact_producing_cases_or_checkpoint_mapping_recovered"]
    low = next(row for row in original["rows"] if row["id"] == "rule_macro+sac")
    assert low["metrics"]["tracking_rmse_kw"] == {"mean": 229.0, "sd": 18.0}
    high = next(row for row in original["rows"] if row["id"] == "sac_macro+sac")
    assert high["metrics"]["tracking_rmse_kw"] == {"mean": 780.0, "sd": 626.0}


def test_new_scope_uses_only_existing_matching_sac_and_classical_controls():
    assert len(ROW_SPECS) == 7
    assert all("ppo" not in row["id"] and "llm" not in row["id"] for row in ROW_SPECS)
    assert MODEL_SPECS["low"]["selected_timesteps"] == 250002
    assert MODEL_SPECS["high"]["selected_timesteps"] == 8004
    assert MODEL_SPECS["low"]["sha256"] == (
        "5940bf534f1ff5222c5eac1554ba7a0108951df864e6a28e220b45a277a09024"
    )


def test_case_and_gate_scope_is_not_the_economic_allocation_task():
    assert PANEL_SPEC["seeds"] == [100, 101, 102, 103, 104]
    assert PANEL_SPEC["start_day"] == 0
    assert PANEL_SPEC["regd_load_bias"] == 0
    assert PANEL_SPEC["service_fast_ticks"] == 17280
    assert PANEL_SPEC["service_macro_calls"] == 96
    assert PANEL_SPEC["start_day_was_in_training"]
    assert SERVICE_GATE_SPEC["capacity_admission_floor"] is None
    assert SERVICE_GATE_SPEC["credit_guard"] is None
    assert SERVICE_GATE_SPEC["service_hardware_action_override"] is None
