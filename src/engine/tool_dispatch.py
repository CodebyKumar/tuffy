"""Parses and executes one ReAct tool call. The one rule that matters here:
NOTHING a tool does is allowed to escape this module as a raw exception —
every failure becomes a ToolExecutionError so a buggy tool (including a
third-party MCP tool the developer doesn't control) degrades to a normal
failed-observation the model can see and react to, instead of killing the
whole Tuffy process.

The previous implementation only caught TypeError here; any other exception
type (ValueError, KeyError, a network error from an MCP tool, ...) propagated
all the way past the turn loop and crashed the session. That gap is the
reason this module exists as its own file with its own test coverage."""

import inspect
import json

from src.engine.errors import ToolExecutionError
from src.engine.events import ToolCall, ToolResult
from src.tools.registry import registry
from src.vision import IMAGE_SENTINEL


def parse_tool_call(tool_call_json: str) -> tuple[str, dict, str]:
    """Returns (function_name, arguments, thought). Raises ToolExecutionError
    if the JSON is malformed or the tool name is missing/placeholder."""
    try:
        tool_info = json.loads(tool_call_json.strip())
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        raise ToolExecutionError(f"tool call is not valid JSON ({e})")

    function_name = tool_info.get("name")
    function_args = tool_info.get("arguments", {}) or {}
    thought = str(tool_info.get("thought", "")).strip()

    if not function_name or function_name in ("tool_name", "exact_tool_name"):
        raise ToolExecutionError("no real tool name given — use an exact name from the TOOLS list")

    return function_name, function_args, thought


def call_signature(function_name: str, function_args: dict) -> str:
    """Canonical (name, sorted-args-json) key for exact-repeat detection."""
    return function_name, json.dumps(function_args, sort_keys=True)


def execute(function_name: str, function_args: dict, thought: str) -> tuple[list, str]:
    """Runs one tool call. Returns (events, tool_output). Raises
    ToolExecutionError — never any other exception type — on any failure,
    whether that's a validation problem (unknown tool, missing args) or the
    tool function itself misbehaving in any way."""
    events = [ToolCall(name=function_name, arguments=function_args, thought=thought)]

    if function_name not in registry.functions:
        raise ToolExecutionError(
            f"tool '{function_name}' does not exist. Available: {list(registry.functions.keys())}",
            tool_name=function_name,
        )

    missing = [arg for arg in registry.required_args(function_name) if arg not in function_args]
    if missing:
        raise ToolExecutionError(
            f"missing required argument(s) {missing} for tool '{function_name}'",
            tool_name=function_name,
        )

    func = registry.functions[function_name]
    sig = inspect.signature(func)
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    sanitized_args = (
        function_args if has_kwargs
        else {k: v for k, v in function_args.items() if k in sig.parameters}
    )

    try:
        tool_output = func(**sanitized_args)
    except Exception as e:
        # Catch-all is deliberate: a tool raising ANY exception — a bad
        # argument value that passes the signature check but fails inside,
        # a network error from an MCP-backed tool, a bug in tool code we
        # don't control — must come back as a failed observation the model
        # can see and retry from, not an unhandled exception that unwinds
        # past the engine and kills the session.
        raise ToolExecutionError(f"'{function_name}' failed: {e}", tool_name=function_name)

    if not isinstance(tool_output, str):
        # A tool returning something other than str (a bug in the tool) is
        # the same class of problem — surface it as a failed call rather
        # than let a non-string propagate into message content downstream.
        raise ToolExecutionError(
            f"'{function_name}' returned {type(tool_output).__name__}, expected str",
            tool_name=function_name,
        )

    if tool_output.startswith(IMAGE_SENTINEL):
        image_path = tool_output[len(IMAGE_SENTINEL):].partition("\n")[0]
        shown = f"(image attached, saved at {image_path})"
    else:
        shown = tool_output
    events.append(ToolResult(name=function_name, result=shown, ok=True))

    return events, tool_output
