import copy

import pytest
from register_study import validate_split
from study import ROOT, read_json


def test_new_windows_accept_exact_two_day_separation():
    split = read_json(ROOT / "split.json")
    validate_split(split, [91001, 91002, 92001, 92002, 92003])
    prior = split["previously_consumed_days"]
    for day in (96, 188, 198, 218, 303, 314):
        assert min(abs(day - old) for old in prior) == 2
    days = split["new_development_days"] + split["new_final_days"]
    assert (
        min(abs(a - b) for index, a in enumerate(days) for b in days[index + 1 :]) == 10
    )


def test_new_window_guards_reject_leakage_and_seed_collisions():
    original = read_json(ROOT / "split.json")
    altered = copy.deepcopy(original)
    altered["new_final_days"][0] = 315
    with pytest.raises(ValueError, match="prior exposure"):
        validate_split(altered, [])
    altered = copy.deepcopy(original)
    altered["new_final_seeds"][0] = 93001
    with pytest.raises(ValueError, match="namespaces collide"):
        validate_split(altered, [])
