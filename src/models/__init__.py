"""Registers every model Tuffy can load. Model cards live in src/models/configs/
(local.py for llama.cpp/gguf weights, api.py for API providers) - add a new
model by calling registry.register(...) in the appropriate file; it becomes
available to /models automatically, no other wiring needed.
"""

import src.models.configs.local  # noqa: F401 - registers local gguf models as a side effect of import
import src.models.configs.api  # noqa: F401 - registers API-provider models as a side effect of import

# First-run default, used only until the user sets one with '/models default
# <id>' (persisted in .tuffy/settings.json - see src/settings.py). Can name
# either a local (llama_cpp) or API (openai_compatible) model; main.py reads
# each model card's own 'provider' field to load it correctly either way, so
# there's nothing to configure differently based on the model's type.
DEFAULT_MODEL = "qwen3vl-2b-instruct-q4km"

# Safety-net local model main.py falls back to at startup if the resolved
# default fails to load (e.g. an API model chosen as default but its API key
# env var isn't set, or the API is unreachable) - keeps Tuffy working fully
# offline with zero configuration regardless of what the user has set as
# their default.
FALLBACK_MODEL = "qwen3vl-2b-instruct-q4km"
