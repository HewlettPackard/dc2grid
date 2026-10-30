from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import sys
import tomllib
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

from .config import contained_path
from .market import MARKETS, Bid, Trade, advance_clock, epoch_seconds, timestamp

UNIT_ID = "C2G_DC"


def require_backend(root: Path) -> None:
    if not (root / "assume_env" / "create_world.py").is_file():
        raise FileNotFoundError(f"3L checkout is missing: {root}")
    missing = [
        name for name in ("assume", "mango") if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ModuleNotFoundError(
            f"Missing 3L runtime dependencies: {', '.join(missing)}. "
            "Importing/staging the 3L checkout does not install ASSUME or mango. "
            "Use an environment provisioned for 3L; no packages were installed."
        )
    with (root / "uv.lock").open("rb") as handle:
        locked = {
            entry["name"]: entry["version"] for entry in tomllib.load(handle)["package"]
        }
    for name in ("assume-framework", "mango-agents"):
        if metadata.version(name) != locked[name]:
            raise RuntimeError(
                f"{name} must match the 3L lockfile version {locked[name]}."
            )


@contextmanager
def local_directory(directory: Path):
    directory = contained_path(directory)
    previous = Path.cwd()
    previous_bytecode = sys.dont_write_bytecode
    variables = {
        name: os.environ.get(name)
        for name in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME")
    }
    os.chdir(directory)
    try:
        sys.dont_write_bytecode = True
        os.environ.update({name: str(directory / "cache") for name in variables})
        yield
    finally:
        os.chdir(previous)
        sys.dont_write_bytecode = previous_bytecode
        for name, value in variables.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def import_three_l(root: Path, name: str = "assume_env.create_world"):
    package = name.partition(".")[0]
    existing = sys.modules.get(package)
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(
        root
    ):
        raise RuntimeError(f"A different {package} checkout is already imported.")
    source = root / "dc_env" if package == "policies" else root
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"3L import requires unavailable module {exc.name!r}. "
            "Provision the dependencies declared by 3L before running live markets."
        ) from exc


class ThreeLBackend:
    def __init__(self, root, config, ledger, output_dir, end_time, seed):
        self.root, self.config, self.ledger = root, deepcopy(config), ledger
        self.output_dir = contained_path(output_dir)
        self.end_time, self.seed = timestamp(end_time), int(seed)
        self.world = None
        self.activated = False
        self.callback_error: Exception | None = None
        self.bid_sequence = 0
        self.submitted: dict[str, Bid] = {}
        self.clearing_events: list[dict] = []

    def start(self, controllers) -> None:
        require_backend(self.root)
        self.controllers = controllers
        self.output_dir.mkdir(parents=True, exist_ok=False)
        with local_directory(self.output_dir):
            helpers = import_three_l(self.root)
            self._build_world(helpers)
            self._register_dc(helpers)
            self._activate()

    def _build_world(self, helpers) -> None:
        from assume import World
        from assume.common.utils import set_random_seed

        from .assume_adapters import RecordedORDC, RecordedPayAsClear

        self._validate_inputs()
        self.config["sim"]["outputs_dir"] = str(self.output_dir / "market_csv")
        self.config["sim"]["database_uri"] = ""
        self.world = World(
            export_csv_path=self.config["sim"]["outputs_dir"], database_uri=""
        )
        helpers.register_strategies(self.world)
        self.world.clearing_mechanisms["pay_as_clear_ordc"] = partial(
            RecordedORDC, recorder=self
        )
        self.world.clearing_mechanisms["pay_as_clear"] = partial(
            RecordedPayAsClear, recorder=self
        )
        helpers._apply_reposition_premiums(self.config)
        set_random_seed(self.seed)
        helpers.load_scenario(self.world, self.config)
        helpers.attach_forecast_profiles(self.world, self.config)
        helpers.add_arbitrage_unit(self.world, self.config)
        helpers.add_system_imbalance_unit(self.world, self.config)
        if set(self.world.markets) != set(MARKETS):
            raise ValueError(
                "The selected 3L scenario must contain EOM, IDM and BM only."
            )

    def _validate_inputs(self) -> None:
        scenario = (
            Path(self.config["sim"]["inputs_path"]) / self.config["sim"]["scenario"]
        )
        with (scenario / "config.yaml").open(encoding="utf-8") as handle:
            calendar = yaml.safe_load(handle)[self.config["sim"]["study_case"]]
        if calendar.get("seed") not in (None, self.seed):
            raise ValueError(
                "Experiment seed must agree with the non-null 3L calendar seed."
            )
        for name in (
            "powerplant_units.csv",
            "demand_units.csv",
            "storage_units.csv",
            "availability_df.csv",
            "availability_id_df.csv",
            "availability_realized_df.csv",
            "fuel_prices_df.csv",
            "demand_df.csv",
            "forecasts_df.csv",
            "system_imbalance_df.csv",
            "temperature_df.csv",
        ):
            if not (scenario / name).is_file():
                raise FileNotFoundError(
                    f"Required 3L input is missing: {scenario / name}"
                )
        for helper in ("idm_arbitrage", "system_imbalance"):
            if not self.config["helpers"][helper]["enable"]:
                raise ValueError(f"The calibrated Ireland setup requires {helper}.")

    def _register_dc(self, helpers) -> None:
        from .assume_adapters import C2GBiddingStrategy, C2GDemandUnit

        self.world.bidding_strategies["c2g_three_level"] = C2GBiddingStrategy
        self.world.unit_types["c2g_dc"] = C2GDemandUnit
        self.world.add_unit_operator(UNIT_ID)
        self.world.add_unit(
            id=UNIT_ID,
            unit_type="c2g_dc",
            unit_operator_id=UNIT_ID,
            unit_params={
                "bridge": self,
                "node": self.config["helpers"]["idm_arbitrage"]["node"],
                "bidding_strategies": {name: "c2g_three_level" for name in MARKETS},
            },
            forecaster=helpers._build_forecaster(self.config),
        )

    def _activate(self) -> None:
        if self.world.distributed_role is not None:
            raise ValueError("This adapter supports one local ASSUME world.")
        self.world._validate_setup()
        self.world.clock.set_time(epoch_seconds(self.world.start) - 1)

        async def activate():
            await self.world.container.start()
            self.activated = True
            self.world.container.on_ready()
            await self._drain()

        self.world.loop.run_until_complete(activate())

    async def _drain(self) -> None:
        from mango.util.termination_detection import tasks_complete_or_sleeping

        await tasks_complete_or_sleeping(self.world.container)
        if self.callback_error is not None:
            raise RuntimeError(
                "The C2G market callback failed."
            ) from self.callback_error

    def advance_to(self, when) -> None:
        if not self.activated:
            raise RuntimeError("Start the market backend before advancing it.")
        with local_directory(self.output_dir):
            self.world.loop.run_until_complete(
                advance_clock(self.world.clock, epoch_seconds(when), self._drain)
            )

    def orders(self, market_config, products) -> list[dict]:
        try:
            return self._orders(market_config, products)
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            self.callback_error = exc
            raise

    def _orders(self, market_config, products) -> list[dict]:
        now = pd.Timestamp(self.world.clock.time, unit="s")
        future = []
        for start, end, only_hours in products:
            if only_hours is not None:
                raise ValueError(
                    "C2G supports continuous delivery products, not only_hours."
                )
            if timestamp(start) >= now and timestamp(end) <= self.end_time:
                future.append((timestamp(start), timestamp(end), None))
        bids = self.controllers.bid(
            market_config.market_id,
            future,
            float(market_config.maximum_bid_price),
            float(market_config.minimum_bid_price),
            now,
        )
        orders = []
        for bid in bids:
            if abs(bid.purchase_mw) < 1e-9:
                continue
            self.bid_sequence += 1
            bid_id = f"{UNIT_ID}_{self.bid_sequence}"
            self.submitted[bid_id] = bid
            orders.append(
                {
                    "start_time": bid.start.to_pydatetime(),
                    "end_time": bid.end.to_pydatetime(),
                    "only_hours": None,
                    "agent_id": UNIT_ID,
                    "unit_id": UNIT_ID,
                    "bid_type": "SB",
                    "bid_id": bid_id,
                    "volume": -bid.purchase_mw,
                    "price": bid.price_eur_mwh,
                    "node": self.config["helpers"]["idm_arbitrage"]["node"],
                }
            )
        return orders

    def capture(self, market, result) -> None:
        try:
            self._capture(market, result)
        except (ValueError, KeyError, TypeError) as exc:
            self.callback_error = exc
            raise

    def _capture(self, market, result) -> None:
        accepted, rejected, metadata_rows, _ = result
        now = pd.Timestamp(self.world.clock.time, unit="s")
        for row in metadata_rows:
            self.ledger.record_price(
                market, row["product_start"], row["product_end"], row["price"], now
            )
        fills = [order for order in accepted if order["unit_id"] == UNIT_ID]
        for order in fills:
            trade = Trade(
                market,
                order["bid_id"],
                timestamp(order["start_time"]),
                timestamp(order["end_time"]),
                -float(order["accepted_volume"]),
                float(order["accepted_price"]),
                now,
            )
            if self.ledger.record_trade(trade):
                self.controllers.on_fill(self.submitted[trade.bid_id], trade)
        self.clearing_events.append(
            {
                "market": market,
                "cleared_at": now.isoformat(),
                "accepted_c2g_orders": len(fills),
                "rejected_c2g_orders": sum(
                    order["unit_id"] == UNIT_ID for order in rejected
                ),
                "products": len(metadata_rows),
            }
        )

    def close(self) -> None:
        if self.world is None:
            return
        world = self.world
        try:
            with local_directory(self.output_dir):
                if self.activated:
                    self.activated = False
                    try:
                        world.loop.run_until_complete(world.output_role.store_dfs())
                    finally:
                        world.loop.run_until_complete(world.container.shutdown())
        finally:
            world.loop.close()
            asyncio.set_event_loop(None)
            self.world = None
