from typing import Literal

SimulationMode = Literal["physical_v1", "legacy_v0"]
SIMULATION_MODES = ("physical_v1", "legacy_v0")


def simulation_mode(value: str) -> SimulationMode:
    if value == "physical_v1":
        return "physical_v1"
    if value == "legacy_v0":
        return "legacy_v0"
    raise ValueError(
        f"Unknown simulation_mode {value!r}; choose from {SIMULATION_MODES}"
    )
