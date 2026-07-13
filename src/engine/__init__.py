"""Provider-agnostic turn engine: the ReAct loop, tool dispatch, and
stream parsing, expressed as a flat stream of typed events (see events.py).
Nothing in this package talks to a terminal, a callback, or a Session — it
only consumes an LLMProvider and a message list, and produces TurnEvents."""
