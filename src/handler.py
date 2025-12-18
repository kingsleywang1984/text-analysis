from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pydantic import ValidationError

from src.config import AppConfig, load_config
from src.llm.factory import create_llm_client
from src.logging_utils import log_error, log_info
from src.models.schemas import AnalyzeRequest
from src.pipeline.analyze import analyze_request


@dataclass(frozen=True, slots=True)
class HttpResponse:
    statusCode: int
    headers: Dict[str, str]
    body: str

    def to_dict(self) -> Dict[str, Any]:
        return {"statusCode": self.statusCode, "headers": self.headers, "body": self.body}


def _json_response(status: int, payload: Dict[str, Any]) -> HttpResponse:
    return HttpResponse(
        statusCode=status,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Cache-Control": "no-store",
        },
        body=json.dumps(payload, ensure_ascii=False),
    )


def _extract_body(event: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (body_text, error_message).
    Supports API Gateway REST (v1) and HTTP API (v2) proxy events.
    """
    body = event.get("body")
    if body is None:
        return None, "Missing request body"
    if not isinstance(body, str):
        return None, "Request body must be a string"

    if event.get("isBase64Encoded") is True:
        try:
            raw = base64.b64decode(body)
            return raw.decode("utf-8"), None
        except Exception:
            return None, "Invalid base64-encoded body"

    return body, None


def _extract_method_path(event: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    # HTTP API v2
    rc = event.get("requestContext") or {}
    http = rc.get("http") or {}
    method = http.get("method")
    path = http.get("path")
    if method and path:
        return str(method).upper(), str(path)

    # REST API v1
    method = event.get("httpMethod")
    path = event.get("path")
    if method and path:
        return str(method).upper(), str(path)

    return None, None


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda entrypoint (API Gateway proxy integration).

    Route: POST /analyze
    """
    start = time.perf_counter()
    log_info("handler.event_received", has_context=bool(context))
    method, path = _extract_method_path(event)
    if method and path and not (method == "POST" and path.endswith("/analyze")):
        return _json_response(404, {"error": "Not Found"}).to_dict()

    body_text, err = _extract_body(event)
    if err:
        return _json_response(400, {"error": err}).to_dict()

    try:
        payload = json.loads(body_text or "")
    except Exception:
        return _json_response(400, {"error": "Request body must be valid JSON"}).to_dict()

    try:
        req = AnalyzeRequest.model_validate(payload)
    except ValidationError as e:
        return _json_response(
            400,
            {"error": "Invalid request schema", "details": json.loads(e.json())},
        ).to_dict()

    try:
        cfg: AppConfig = load_config()
        log_info(
            "handler.config_loaded",
            elapsed_seconds=round(time.perf_counter() - start, 4),
        )
        llm = create_llm_client(cfg)
        resp_start = time.perf_counter()
        resp = analyze_request(req, cfg, llm=llm)
        elapsed = time.perf_counter() - resp_start
        total = time.perf_counter() - start
        log_info(
            "handler.analyze_request_completed",
            analyze_duration_seconds=round(elapsed, 4),
            total_duration_seconds=round(total, 4),
        )
        # Pydantic models can be dumped to JSON-ready dict
        return _json_response(200, resp.model_dump(mode="json")).to_dict()
    except Exception as e:
        log_error("handler.exception", error_type=type(e).__name__, error=str(e))
        # Fail closed: do not leak internal exception details by default.
        return _json_response(500, {"error": "Internal Server Error"}).to_dict()
