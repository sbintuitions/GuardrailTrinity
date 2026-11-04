# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Pytest path setup: make data/, eval/, and training/ importable from tests/."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("data", "eval", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
