"""Pytest configuration for shared fixtures and environment tweaks."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_TEST = PROJECT_ROOT / ".env.test"

# Ensure imports like `from src...` resolve without relying on installer metadata.
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

if ENV_TEST.exists():
    load_dotenv(ENV_TEST, override=True)


