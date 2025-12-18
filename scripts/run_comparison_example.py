#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

ENV_FILE = ROOT_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)

from src.config import load_config
from src.llm.client import FakeLLMClient
from src.llm.factory import create_llm_client
from src.models.schemas import AnalyzeRequest
from src.pipeline.analyze import analyze_request

DATA_FILE = ROOT_DIR / "data" / "input_comparison_example.json"


def main() -> None:
    try:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Missing input file: {DATA_FILE}")

        payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        cfg = load_config()
        req = AnalyzeRequest.model_validate(payload)

        # LLM-first (if configured) with deterministic fallback.
        # For local demo convenience: if no provider configured, use FakeLLMClient.
        llm = create_llm_client(cfg) or FakeLLMClient()

        resp = analyze_request(req, cfg, llm=llm)
        print(resp.model_dump_json(indent=2, ensure_ascii=False))
    except Exception as e:
        print("ERROR running comparison example:", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()


