"""Structured logging + per-request/per-agent-node observability.

Two things live here:
  - `configure_logging()`: JSON-formatted stdlib logging, so log lines are
    machine-parseable (grep/jq-able, or shippable to a log aggregator)
    instead of free text.
  - `RequestLoggingMiddleware` / `log_node_timing`: emit one structured line
    per HTTP request and per agent-graph node, with latency and outcome.

LangGraph/LangChain call-level tracing is handled separately, through
LangSmith's own env vars (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY,
LANGCHAIN_PROJECT - see .env.example): LangChain picks those up automatically
with no code needed here. What LangSmith *doesn't* see is this app's own
graph-node boundaries (retrieve vs. grade vs. graph_augment vs. generate) -
that's what `log_node_timing` adds, applied to each node in
`app.services.qa_agent`.
"""
import functools
import json
import logging
import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """One structured JSON log line per request: method, path, status, and
    latency, tagged with a request id that's also echoed back as a response
    header so a caller can correlate their request with server-side logs."""

    def __init__(self, app):
        super().__init__(app)
        self._logger = logging.getLogger("app.request")

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        self._logger.info(
            "request",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        return response


def log_node_timing(node_name: str) -> Callable:
    """Decorator for LangGraph node functions: logs how long each node in the
    agent graph took, and whether it raised."""

    def decorator(fn: Callable) -> Callable:
        logger = logging.getLogger("app.agent")

        @functools.wraps(fn)
        def wrapper(state):
            start = time.monotonic()
            try:
                result = fn(state)
                duration_ms = round((time.monotonic() - start) * 1000, 2)
                logger.info(
                    "agent_node",
                    extra={"extra_fields": {"node": node_name, "duration_ms": duration_ms, "status": "ok"}},
                )
                return result
            except Exception:
                duration_ms = round((time.monotonic() - start) * 1000, 2)
                logger.exception(
                    "agent_node_failed",
                    extra={"extra_fields": {"node": node_name, "duration_ms": duration_ms, "status": "error"}},
                )
                raise

        return wrapper

    return decorator
