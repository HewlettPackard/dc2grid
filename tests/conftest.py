"""
conftest.py — pytest configuration for C2G-Macro test suite.
Ensures all tests run from the project root so relative data paths resolve.
"""
import os
import pytest

@pytest.fixture(autouse=True)
def set_project_root(monkeypatch):
    """Change CWD to project root for every test."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(project_root)
