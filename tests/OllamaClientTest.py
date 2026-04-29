import json

import src.llm.OllamaClient as ollama_module
from src.llm.OllamaClient import OllamaClient


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeResponse:
    def __init__(self, content: str):
        self.message = FakeMessage(content=content)


def test_invoke_text_logs_raw_llm_call(tmp_path, monkeypatch):
    log_path = tmp_path / "llm_calls.jsonl"

    def fake_chat(**kwargs):
        return FakeResponse("Hello from the model")

    monkeypatch.setattr(ollama_module, "chat", fake_chat)

    client = OllamaClient(model="test-model", log_path=log_path)
    messages = [{"role": "user", "content": "Say hello"}]

    result = client.invoke_text(messages=messages, stream=False)

    assert result == "Hello from the model"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["call_type"] == "text"
    assert entry["model"] == "test-model"
    assert entry["messages"] == messages
    assert entry["raw_response"] == "Hello from the model"
    assert entry["options"] == {"temperature": 0, "top_p": 0.1}


def test_invoke_json_logs_raw_and_parsed_llm_call(tmp_path, monkeypatch):
    log_path = tmp_path / "llm_calls.jsonl"
    raw_response = '{"accepted": true, "issues": []}'

    def fake_chat(**kwargs):
        return FakeResponse(raw_response)

    monkeypatch.setattr(ollama_module, "chat", fake_chat)

    client = OllamaClient(model="test-model", log_path=log_path)
    messages = [{"role": "user", "content": "Return JSON"}]
    schema = {"type": "object"}

    result = client.invoke_json(messages=messages, stream=False, schema=schema)

    assert result == {"accepted": True, "issues": []}

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["call_type"] == "json"
    assert entry["model"] == "test-model"
    assert entry["messages"] == messages
    assert entry["schema"] == schema
    assert entry["raw_response"] == raw_response
    assert entry["parsed_response"] == {"accepted": True, "issues": []}


def test_invoke_json_logs_parse_error_before_reraising(tmp_path, monkeypatch):
    log_path = tmp_path / "llm_calls.jsonl"

    def fake_chat(**kwargs):
        return FakeResponse("not json")

    monkeypatch.setattr(ollama_module, "chat", fake_chat)

    client = OllamaClient(model="test-model", log_path=log_path)

    try:
        client.invoke_json(
            messages=[{"role": "user", "content": "Return JSON"}],
            stream=False,
            schema={"type": "object"}
        )
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Expected JSONDecodeError")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["raw_response"] == "not json"
    assert "parse_error" in entry
