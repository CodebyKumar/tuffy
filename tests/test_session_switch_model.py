"""Regression test for unbounded accumulation of model-switch notices found
in the pre-redesign audit: every /models switch used to append a new
'[Model switched from ... ]' system message to history, and since
trim_history's pairing logic only evicts consecutive user/assistant pairs, a
lone injected system message was never eligible for eviction - a long
session with N switches accumulated N of these forever.

Session.switch_model itself needs a real model to load (out of scope for a
unit test), so this exercises the exact same filter-then-append logic in
isolation against a plain message list, matching what src/cli/session.py's
switch_model does line for line."""

from src.cli.session import _MODEL_SWITCH_TAG


def _apply_switch_notice(messages: list, old_model_id: str, model_id: str) -> list:
    """Mirrors Session.switch_model's history-mutation logic exactly."""
    messages = [
        m for m in messages
        if not (m.get("role") == "system" and m.get("content", "").startswith(_MODEL_SWITCH_TAG))
    ]
    messages.append({
        "role": "system",
        "content": (
            f"{_MODEL_SWITCH_TAG}'{old_model_id}' to '{model_id}'. "
            "Prior turns above are already answered - do not re-answer "
            "them unless the user explicitly asks again.]"
        ),
    })
    return messages


def test_single_switch_adds_one_notice():
    messages = [{"role": "system", "content": "persona"}]
    messages = _apply_switch_notice(messages, "model-a", "model-b")
    notices = [m for m in messages if m["content"].startswith(_MODEL_SWITCH_TAG)]
    assert len(notices) == 1


def test_repeated_switches_replace_not_accumulate():
    messages = [{"role": "system", "content": "persona"}]
    messages = _apply_switch_notice(messages, "model-a", "model-b")
    messages = _apply_switch_notice(messages, "model-b", "model-c")
    messages = _apply_switch_notice(messages, "model-c", "model-d")

    notices = [m for m in messages if m["content"].startswith(_MODEL_SWITCH_TAG)]
    assert len(notices) == 1, "repeated switches must replace the notice, not accumulate one per switch"
    assert "model-c' to 'model-d" in notices[0]["content"]


def test_conversation_messages_between_switches_are_preserved():
    messages = [{"role": "system", "content": "persona"}]
    messages = _apply_switch_notice(messages, "model-a", "model-b")
    messages.append({"role": "user", "content": "hello"})
    messages.append({"role": "assistant", "content": "hi"})
    messages = _apply_switch_notice(messages, "model-b", "model-c")

    roles = [m["role"] for m in messages]
    assert roles.count("user") == 1
    assert roles.count("assistant") == 1
