"""DeepSeek provider: selected via LLM_PROVIDER=deepseek, reads DEEPSEEK_*
env vars, and R1-style reasoning (<think> blocks in content, or the separate
reasoning_content field) never reaches the agent loop or the speaker. No
network calls -- httpx is mocked.
"""

import json

import httpx
import pytest

from app.llm import client as llm


def _client(handler):
    c = llm.DeepSeekClient(llm.LLMConfig(base_url="http://example.test/v1", api_key="k", model="m"))
    c._http = httpx.Client(base_url="http://example.test/v1", transport=httpx.MockTransport(handler))
    return c


def _reply(message, finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})


# ---- factory / env ---------------------------------------------------------


def test_deepseek_selected_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    with llm.get_llm_client() as c:
        assert isinstance(c, llm.DeepSeekClient)
        assert isinstance(c, llm.BaseLLMClient)
        assert c.config.base_url == "https://api.deepseek.com/v1"
        assert c.config.model == "deepseek-reasoner"
        assert c._http.headers["Authorization"] == "Bearer sk-test"


def test_deepseek_defaults_to_chat_model_and_warns_without_key(monkeypatch, caplog):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    with caplog.at_level("WARNING"), llm.DeepSeekClient() as c:
        assert c.config.model == "deepseek-chat"
    assert "DEEPSEEK_API_KEY" in caplog.text


def test_deepseek_listed_in_providers():
    assert llm.LLM_PROVIDERS["deepseek"] is llm.DeepSeekClient


# ---- thinking-tag handling -------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("<think>let me reason</think>Revenue was 405 thousand.", "Revenue was 405 thousand."),
        ("<think>\nmulti\nline\n</think>\n\nAnswer here.", "Answer here."),
        ("<think>a</think>First.<think>b</think> Second.", "First. Second."),
        ("<think>cut off by max_tokens", ""),
        ("No thinking at all.", "No thinking at all."),
    ],
)
def test_strip_thinking(raw, expected):
    assert llm._strip_thinking(raw) == expected


def test_complete_strips_think_tags_from_text_and_raw_message():
    c = _client(lambda req: _reply({"role": "assistant",
                                    "content": "<think>405200 -> about 405 thousand</think>Revenue was about 405 thousand dollars."}))
    result = c.complete([{"role": "user", "content": "q"}])
    assert result.text == "Revenue was about 405 thousand dollars."
    # The agent loop reads the raw message, not Completion.text.
    assert result.raw["choices"][0]["message"]["content"] == "Revenue was about 405 thousand dollars."
    assert "<think>" not in json.dumps(result.raw)


def test_complete_drops_reasoning_content_from_raw_message():
    c = _client(lambda req: _reply({"role": "assistant",
                                    "reasoning_content": "The user wants Q3 revenue...",
                                    "content": "Revenue was about 405 thousand dollars."}))
    result = c.complete([{"role": "user", "content": "q"}])
    assert result.text == "Revenue was about 405 thousand dollars."
    # DeepSeek returns 400 if reasoning_content is echoed back in history.
    assert "reasoning_content" not in result.raw["choices"][0]["message"]


def test_tool_call_with_reasoning_keeps_empty_content():
    message = {
        "role": "assistant",
        "reasoning_content": "I should call the tool.",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "query_financials", "arguments": "{}"}}],
    }
    c = _client(lambda req: _reply(message, finish_reason="tool_calls"))
    result = c.complete([{"role": "user", "content": "q"}], tools=[{"type": "function", "function": {"name": "query_financials", "parameters": {}}}])
    raw_message = result.raw["choices"][0]["message"]
    assert raw_message["tool_calls"][0]["function"]["name"] == "query_financials"
    assert raw_message["content"] == ""          # reasoning must not masquerade as an answer
    assert "reasoning_content" not in raw_message
    assert result.text == ""


def test_reasoning_only_reply_falls_back_to_reasoning_text():
    c = _client(lambda req: _reply({"role": "assistant", "content": "", "reasoning_content": "Only reasoning came back."}))
    assert c.complete([{"role": "user", "content": "q"}]).text == "Only reasoning came back."


def test_stream_measured_strips_think_tags():
    body = "".join(
        f"data: {json.dumps({'choices': [{'delta': {'content': part}}]})}\n\n"
        for part in ["<think>hmm", " more</think>", "Final ", "answer."]
    ) + "data: [DONE]\n\n"
    c = _client(lambda req: httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"}))
    result, _ = c.stream_measured([{"role": "user", "content": "q"}])
    assert result.text == "Final answer."
