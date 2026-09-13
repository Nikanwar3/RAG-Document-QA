import json
import logging

import pytest

from app.observability import JsonFormatter, log_node_timing


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", None, None)
    record.extra_fields = {"foo": "bar"}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["foo"] == "bar"


def test_json_formatter_without_extra_fields_still_serializes():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", None, None)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello"


def test_log_node_timing_passes_through_result_and_logs_ok(caplog):
    @log_node_timing("dummy")
    def node(state):
        return {"ok": True, "state": state}

    with caplog.at_level(logging.INFO, logger="app.agent"):
        result = node({"x": 1})

    assert result == {"ok": True, "state": {"x": 1}}
    ok_records = [r for r in caplog.records if r.message == "agent_node"]
    assert len(ok_records) == 1
    assert ok_records[0].extra_fields["node"] == "dummy"
    assert ok_records[0].extra_fields["status"] == "ok"


def test_log_node_timing_reraises_and_logs_failure(caplog):
    @log_node_timing("dummy")
    def node(state):
        raise ValueError("boom")

    with caplog.at_level(logging.INFO, logger="app.agent"), pytest.raises(ValueError, match="boom"):
        node({})

    failed_records = [r for r in caplog.records if r.message == "agent_node_failed"]
    assert len(failed_records) == 1
    assert failed_records[0].extra_fields["status"] == "error"


@pytest.mark.asyncio
async def test_request_logging_middleware_tags_response_with_request_id(client):
    response = await client.get("/health")

    assert "X-Request-ID" in response.headers
