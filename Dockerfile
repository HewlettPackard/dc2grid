# syntax=docker/dockerfile:1

# =============================================================================
# C2G-Bench — Hierarchical AI Orchestration for Grid-Interactive Data Centers
# NeurIPS 2026 Benchmark — reproducible container image.
#
# The project targets Python 3.11 (pinned to ==3.11.* in pyproject.toml) and is
# managed with `uv`. On Linux, pyproject.toml resolves PyTorch from the CUDA 12.4
# wheel index, so the image works on both CPU-only hosts and NVIDIA GPU hosts
# (add `--gpus all` at run time to use the GPU).
#
# Build:   docker build -t c2g-bench .
# Run:     docker run --rm -it c2g-bench            # opens a shell, ready to use
# GPU:     docker run --rm -it --gpus all c2g-bench
# =============================================================================

# uv's official image ships a ready-to-use Python 3.11 + uv toolchain, matching
# the .python-version (3.11.12) pinned by the repo.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# --- uv behaviour tuning ------------------------------------------------------
# UV_LINK_MODE=copy:   avoid hardlink warnings across Docker layer boundaries.
# UV_PROJECT_ENVIRONMENT: install the venv at a fixed, predictable path.
# Note: UV_COMPILE_BYTECODE is intentionally left off. Parallel .pyc compilation
# can exhaust the file-descriptor limit on constrained build hosts; skipping it
# keeps the build portable at the cost of a slightly slower first import.
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

# --- System libraries ---------------------------------------------------------
# git:            some tooling / VCS operations expect it to be present.
# build-essential: fallback compiler for any sdist that lacks a prebuilt wheel.
# libgomp1:       OpenMP runtime required by numpy / scipy / torch kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer (cached) ------------------------------------------------
# Copy only the files needed to resolve dependencies first, so the (slow) install
# layer is reused whenever project source code changes but dependencies do not.
# README.md is copied because pyproject.toml references it via `readme = ...`.
COPY pyproject.toml README.md .python-version ./
COPY c2g_env/__init__.py ./c2g_env/__init__.py
COPY preprocessing/__init__.py ./preprocessing/__init__.py

# Resolve and install all runtime + dev dependencies (pytest, ruff, mypy).
# --no-install-project: install ONLY third-party deps in this cached layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --no-install-project

# --- Application layer --------------------------------------------------------
# Now copy the full repository and install the project itself into the venv.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev

# Put the project virtualenv on PATH so `python`, `pytest`, etc. resolve to it
# without needing the `uv run` prefix.
ENV PATH="/app/.venv/bin:${PATH}"

# --- Default command ----------------------------------------------------------
# Drop the user into an interactive shell inside the environment. From here they
# can run any of the documented commands, e.g.:
#   pytest tests/ -q
#   python train/train_ppo.py scenario=scenario_a market=pjm_dom
#   bash scripts/run_sweep.sh --dry-run
CMD ["bash"]
