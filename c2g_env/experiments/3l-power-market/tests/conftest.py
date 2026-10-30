from copy import deepcopy
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

PACKAGE = "c2g_env.experiments.3l-power-market"


@pytest.fixture(autouse=True)
def local_caches(monkeypatch):
    cache = Path(__file__).resolve().parents[1] / ".runtime" / "cache"
    for name in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(cache))


@pytest.fixture(scope="session")
def modules():
    names = (
        "config",
        "plant",
        "forecast",
        "market",
        "controllers",
        "env",
        "assume_backend",
    )
    return SimpleNamespace(
        **{name: import_module(f"{PACKAGE}.{name}") for name in names}
    )


@pytest.fixture
def config(modules):
    return deepcopy(modules.config.load_config())


@pytest.fixture
def plant(modules, config):
    result = modules.plant.C2GPlant(config["c2g"])
    result.reset(0, 7.0)
    return result


@pytest.fixture(scope="session")
def ireland_data(modules):
    cfg = modules.config.load_config()
    _, _, scenario, _ = modules.config.three_l_config(cfg)
    return modules.market.ThreeLData(scenario)


@pytest.fixture
def tmp_path(modules):
    parent = modules.config.MODULE_ROOT / ".runtime"
    parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="test-", dir=parent) as directory:
        yield Path(directory)
