"""Thin wrapper around an LLMProvider: owns load/unload lifecycle, vision
capability flags, and the non-streaming complete() used by memory's
background jobs. The ReAct loop itself lives in turn_engine.py and does not
belong to this class — unlike the previous LocalAgent, this object has no
status_cb/trace_cb side channels; callers get everything by iterating
turn_engine.run_turn(agent.provider, agent.sampling_params, messages)."""

from src.llm import build_provider


class ModelAgent:
    def __init__(self, model_card: dict):
        self.model_id = model_card["id"]
        self.sampling_params = model_card["sampling_params"]
        self.provider = build_provider(model_card)
        self.provider.load()

    @property
    def supports_vision(self) -> bool:
        return self.provider.supports_vision

    @property
    def vision_disabled_reason(self):
        return self.provider.vision_disabled_reason

    def unload(self):
        self.provider.unload()

    @staticmethod
    def attach_image(user_message: dict, image_data_uri: str) -> dict:
        """Rewrites a plain-text user message into the OpenAI-style multimodal
        content list form, appending an image alongside the existing text.
        image_data_uri is a data: URI (base64) or a plain http(s) URL - both
        llama.cpp's MTMDChatHandler and OpenAI-compatible vision APIs accept
        this same content-block shape."""
        return {
            "role": user_message["role"],
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": user_message["content"]},
            ],
        }

    def complete(self, **kwargs):
        """Non-streaming completion for internal side-tasks (memory
        reflection, session summaries) that shouldn't touch chat history."""
        return self.provider.complete(**kwargs)
