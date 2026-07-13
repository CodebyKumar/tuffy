"""Test doubles for the turn engine's dependencies: a scriptable fake
LLMProvider (so tests control exactly what the model 'says' hop by hop,
without a real model) and small helpers for registering throwaway tools."""

from src.llm.base import LLMProvider, ProviderError


def chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


class FakeProvider(LLMProvider):
    """Replays a scripted sequence of completions. Each item in `script` is
    either a string (split into a few chunks and streamed) or a callable
    exception-raiser, so a test can make the N-th completion fail."""

    def __init__(self, script: list):
        self.model_card = {}
        self._script = list(script)
        self._call_count = 0

    def load(self):
        pass

    def unload(self):
        pass

    def complete(self, **kwargs):
        raise NotImplementedError("FakeProvider only supports stream_completion")

    def stream_completion(self, messages: list, **sampling_params):
        if self._call_count >= len(self._script):
            raise AssertionError(
                f"FakeProvider script exhausted after {self._call_count} calls "
                f"but the engine asked for another completion"
            )
        item = self._script[self._call_count]
        self._call_count += 1

        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, str):
            item = item()

        # Stream in small fixed-size pieces so tag-splitting logic in the
        # parser gets exercised the same way it would with real tokens.
        text = item
        piece_size = 7
        for i in range(0, len(text), piece_size):
            yield chunk(text[i:i + piece_size])
