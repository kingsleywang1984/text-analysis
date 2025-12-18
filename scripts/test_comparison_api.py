#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "input_comparison_example.json"
ANALYZE_URL = "https://pwh7rgqw92.execute-api.ap-southeast-2.amazonaws.com/analyze"


def main() -> None:
    try:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Missing input file: {DATA_FILE}")

        payload: Any = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        baseline_count = len(payload.get("baseline", []))
        comparison_count = len(payload.get("comparison", []) or [])
        say(
            f"Loaded comparison payload from {DATA_FILE} (baseline: {baseline_count}, comparison: {comparison_count})"
        )

        resp = requests.post(
            ANALYZE_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )
        say(f"HTTP {resp.status_code}; request id: {resp.headers.get('x-amzn-requestid', 'n/a')}")

        if resp.status_code != 200:
            say("Response body:")
            say(resp.text)
            resp.raise_for_status()

        parsed = resp.json()
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as exc:
        print("ERROR calling deployed comparison endpoint:", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)


def say(msg: str) -> None:
    print(msg)


if __name__ == "__main__":
    main()
