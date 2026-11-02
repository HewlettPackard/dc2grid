import pytest
from finish_low import bc_reference_index


def test_reference_lookup_uses_semantics_not_historical_id_prefixes():
    rows = [
        {
            "policy": {
                "id": "rule__full__bc_only_1100",
                "family": "bc_only",
                "macro": "rule",
                "training_seed": 1100,
            },
        }
    ]
    assert bc_reference_index({"arbitrary_historical_name": rows}) == {1100: rows}
    with pytest.raises(ValueError, match="Ambiguous"):
        bc_reference_index({"first": rows, "second": rows})
