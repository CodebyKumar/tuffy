"""Any OpenAI-wire-format HTTP API: OpenAI itself, Groq, Together, OpenRouter,
or a local server (vLLM, Ollama's OpenAI-compat endpoint) that speaks the same
JSON shape. One adapter covers all of them via provider_config's base_url +
api_key_env + model_name — no provider-specific SDK, just requests.

The API key is read from the environment variable named in provider_config at
load() time only; it is never written to disk, logged, or echoed back in any
tool output or trace. If it's not already exported in the shell, load() falls
back to reading it from a .env file in the repo root (KEY=value per line) —
a real exported var always takes precedence.
"""

import json
import os

import requests

from src.llm.base import LLMProvider, ProviderError

_DEFAULT_TIMEOUT = 60
_DOTENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")


def _read_dotenv_key(key: str) -> str | None:
    """Reads a single KEY=value out of .env without pulling in a dependency
    or loading the whole file into os.environ — just enough to fall back on
    when the caller hasn't exported the var themselves."""
    if not os.path.isfile(_DOTENV_PATH):
        return None
    with open(_DOTENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            file_key, _, value = line.partition("=")
            if file_key.strip() == key:
                return value.strip().strip("'\"")
    return None


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, model_card: dict):
        super().__init__(model_card)
        cfg = model_card["provider_config"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key_env = cfg["api_key_env"]
        self.model_name = cfg["model_name"]
        self.sampling_params = model_card["sampling_params"]
        self._api_key = None

    def load(self) -> None:
        api_key = os.environ.get(self.api_key_env) or _read_dotenv_key(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable '{self.api_key_env}' is not set — required to use "
                f"model '{self.model_card['id']}'. Export it, or add it to .env in the "
                f"repo root, and try again."
            )
        self._api_key = api_key

    def unload(self) -> None:
        self._api_key = None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list, stream: bool, sampling_params: dict) -> dict:
        # Translate the small set of sampling knobs this codebase actually
        # uses onto OpenAI's chat-completions field names; anything a local
        # model card set that OpenAI's API doesn't understand (mirostat_*,
        # tfs_z, etc.) is silently dropped rather than sent and rejected.
        allowed = {
            "temperature", "top_p", "max_tokens", "presence_penalty",
            "frequency_penalty", "stop", "seed",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        for key in allowed:
            value = sampling_params.get(key)
            if value not in (None, [], ""):
                payload[key] = value
        return payload

    def complete(self, **kwargs) -> dict:
        messages = kwargs.pop("messages")
        sampling_params = {**self.sampling_params, **kwargs}
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stream=False, sampling_params=sampling_params),
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ProviderError(self._describe_error(e)) from e
        return resp.json()

    def stream_completion(self, messages: list, **sampling_params):
        params = sampling_params or self.sampling_params
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, stream=True, sampling_params=params),
                stream=True,
                timeout=_DEFAULT_TIMEOUT,
            )
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data:"):
                    continue
                data = decoded[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                yield chunk
        except requests.exceptions.RequestException as e:
            raise ProviderError(self._describe_error(e)) from e

    def _describe_error(self, e: requests.exceptions.RequestException) -> str:
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            status = e.response.status_code
            if status == 429:
                return (
                    f"'{self.model_card['id']}' is rate-limited (429 from "
                    f"{self.base_url}). Wait a bit or switch models with /models."
                )
            if status in (401, 403):
                return (
                    f"'{self.model_card['id']}' rejected the request ({status}) — "
                    f"check that {self.api_key_env} is a valid key."
                )
            return f"'{self.model_card['id']}' request failed with HTTP {status}."
        if isinstance(e, requests.exceptions.Timeout):
            return f"'{self.model_card['id']}' request timed out after {_DEFAULT_TIMEOUT}s."
        return f"'{self.model_card['id']}' request failed: {e}"
