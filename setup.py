"""Compatibility shim — all metadata lives in ``pyproject.toml`` (PEP 621).

This file is kept so that ``pip install -e .`` works on older pip/setuptools
combinations that don't yet drive editable installs purely from
``pyproject.toml``. Remove once the minimum supported pip is 21.3+.
"""
from setuptools import setup

setup()
