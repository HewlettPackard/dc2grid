"""
C2G-Bench  —  Cloud-to-Grid Macro Benchmark
=============================================
Hierarchical RL environments for grid-interactive 250 MW hyperscale
data centres.

Public API
----------
  C2GFastEnv   — low-level, 5-min timestep, 3 levers (throttle/HVAC/BESS)
  C2GMacroEnv  — high-level, 15-min timestep, wraps C2GFastEnv
"""
from c2g_env.env_low_level  import C2GFastEnv
from c2g_env.env_high_level import C2GMacroEnv

__all__ = ["C2GFastEnv", "C2GMacroEnv"]
__version__ = "0.1.0"
