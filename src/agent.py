"""The agent orchestration layer: drives the ReAct tool-calling loop against
whichever LLMProvider the active model card resolves to (see src/llm/) —
local llama.cpp or an OpenAI-compatible API. All prompt text (persona,
tool-output framing, error messages) lives in src/prompts/ - this module only
orchestrates, and never talks to llama_cpp or an HTTP client directly."""

import json
import re

from src.memory import add_lesson
from src.llm import build_provider
from src.tools.registry import registry
from src.prompts import templates
from src.vision import IMAGE_SENTINEL

_MAX_TOOL_HOPS = 4

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"

# Scripts the model cannot write reliably itself (Greek/Cyrillic through
# Indic through CJK). Symbols, punctuation and emoji are deliberately outside
# these ranges. Text in these scripts is only trusted when it came out of a
# tool (i.e. the translate tool) this turn.
_FOREIGN_SCRIPT_PATTERN = re.compile(r"[Ͱ-῿⺀-퟿]")

# A known small-model failure mode: the reply starts with a leaked chat-role
# token — either the ENTIRE reply is just the bare word (sometimes with
# trailing whitespace/punctuation), or the role word leaks as a prefix
# before the model recovers and continues with real content a line later
# ("user\nI'm Tuffy, ..."). Both come from the same cause: the PROTOCOL
# EXAMPLES block's literal "User:"/"Assistant:" lines pull the next-token
# distribution onto a role-label token right at the start of generation.
# Empty output is the same failure (nothing was said at all). Deliberately
# anchored to the START of the reply and requires a line break or full-string
# match after the role word, so a real answer that merely CONTAINS "user"
# (e.g. "as a user, you can...") is never touched.
_DEGENERATE_PREFIX_PATTERN = re.compile(
    r"^(user|assistant|system)([\s:.,!?]*$|\s*\n)", re.IGNORECASE
)


def _is_degenerate_reply(text: str) -> bool:
    stripped = text.strip()
    return not stripped or bool(_DEGENERATE_PREFIX_PATTERN.match(stripped))


class ToolCallError(Exception):
    """A tool call that couldn't be parsed or executed; carries the tool name
    (when known) so the loop can record a lesson if a retry later succeeds."""

    def __init__(self, message: str, tool_name: str = None):
        super().__init__(message)
        self.tool_name = tool_name


class LocalAgent:
    """Despite the name (kept for compatibility with callers/main.py), this
    now drives ANY provider — local gguf via llama.cpp or a remote API model
    — chosen by the model card's 'provider' field. The ReAct loop, tool
    dispatch, and foreign-script guard below are provider-agnostic; only
    src/llm/*_provider.py contains backend-specific code."""

    def __init__(self, model_card: dict):
        self.model_id = model_card["id"]
        self.sampling_params = model_card["sampling_params"]
        # Optional callable(str) the UI can set to show live status (the
        # model's current thought / which tool is running).
        self.status_cb = None
        # Optional callable(str) for the full ReAct trace (tool calls,
        # arguments, raw results) — set by the CLI only in Ray mode. When
        # None, the agent stays silent about its internal steps; the caller
        # decides what a user is shown, not this module.
        self.trace_cb = None

        self.provider = build_provider(model_card)
        self.provider.load()

    @property
    def supports_vision(self) -> bool:
        return self.provider.supports_vision

    @property
    def vision_disabled_reason(self):
        return self.provider.vision_disabled_reason

    def unload(self):
        """Releases whatever resources the active provider holds (local
        model memory, or just its API credentials) before another model is
        loaded in its place."""
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

    def _status(self, text: str):
        if self.status_cb is not None and text:
            self.status_cb(text)

    def _trace(self, event: str, **data):
        if self.trace_cb is not None:
            self.trace_cb(event, data)

    def complete(self, **kwargs):
        """Non-streaming completion for internal side-tasks (memory
        reflection, session summaries) that shouldn't touch chat history."""
        return self.provider.complete(**kwargs)

    def run_stream(self, messages: list):
        """The ReAct loop: Thought -> Action (<tool_call>) -> Observation ->
        ... -> final Answer, streamed token by token.

        Loops on tool calls until the model gives a plain-text answer or the
        hop budget runs out. Failed calls come back as error observations so
        the model can self-correct; a correction that then succeeds is saved
        as a lesson for future sessions. A final answer containing non-Latin
        script the model wrote itself (rather than relayed from a tool) is
        intercepted and redirected through the translate tool.
        """
        turn_tool_outputs = []  # everything tools returned this turn, for the script guard
        failed_tools = {}       # tool name -> first error message, for lesson capture
        seen_calls = set()      # (name, sorted-args-json) already executed this turn

        for hop in range(_MAX_TOOL_HOPS):
            is_last_hop = hop == _MAX_TOOL_HOPS - 1
            self._status("thinking")

            response_text, is_tool_call, unsourced_foreign, degenerate_start = yield from self._stream_completion(
                messages, sourced_text="".join(turn_tool_outputs)
            )

            if not is_tool_call:
                if degenerate_start and not is_last_hop:
                    # The model's draft collapsed into a leaked chat-role word
                    # ("user") instead of real content. _stream_completion
                    # holds this back before it's ever shown, so nothing
                    # leaked to the user - safe to just ask fresh for a real
                    # answer. Don't add the degenerate draft to history (it
                    # would only prime the retry to repeat it).
                    self._status("retrying")
                    messages.append({"role": "user", "content": templates.degenerate_reply_correction()})
                    continue
                if unsourced_foreign and not is_last_hop:
                    # The model hand-wrote foreign script — unreliable. Ask it
                    # to route through translate and keep looping.
                    self._status("rewriting via translate")
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": templates.foreign_script_correction()})
                    continue
                if degenerate_start and is_last_hop:
                    # Out of hops and still degenerate — force one guaranteed
                    # real answer rather than showing nothing.
                    yield from self._final_answer_guaranteed(messages)
                    return
                if not response_text.strip():
                    # Not flagged degenerate (e.g. a stream that produced
                    # nothing but whitespace, never reaching the holdback
                    # buffer at all) but still empty - treat the same as a
                    # degenerate reply rather than silently ending the turn
                    # with a blank assistant message saved to history.
                    if is_last_hop:
                        yield from self._final_answer_guaranteed(messages)
                    else:
                        self._status("retrying")
                        messages.append({"role": "user", "content": templates.degenerate_reply_correction()})
                        continue
                return

            tool_call_match = _TOOL_CALL_PATTERN.search(response_text)
            messages.append({"role": "assistant", "content": response_text})

            call_signature = self._call_signature(tool_call_match)
            if call_signature is not None and call_signature in seen_calls:
                # Same tool + same arguments already ran this turn: repeating
                # it can't produce new information (a small model will
                # otherwise spend its whole hop budget re-running e.g.
                # get_system_stats). Tell it plainly instead of executing.
                messages.append({
                    "role": "user",
                    "content": templates.repeated_call_blocked(is_last_hop)
                })
                if is_last_hop:
                    yield from self._final_answer_guaranteed(messages)
                    return
                continue

            try:
                tool_output, function_name = self._execute_tool_call(tool_call_match)
            except ToolCallError as e:
                failed_tools.setdefault(e.tool_name or "?", str(e))
                messages.append({
                    "role": "user",
                    "content": templates.tool_call_failed(str(e), is_last_hop)
                })
                if is_last_hop:
                    yield from self._final_answer_guaranteed(messages)
                    return
                continue

            if call_signature is not None:
                seen_calls.add(call_signature)
            turn_tool_outputs.append(tool_output)
            if function_name in failed_tools:
                # Self-correction succeeded: keep the lesson for next time.
                add_lesson(f"{function_name}: earlier call failed ({failed_tools.pop(function_name)[:120]}); corrected call worked")

            if tool_output.startswith(IMAGE_SENTINEL):
                image_path, _, image_data_uri = tool_output[len(IMAGE_SENTINEL):].partition("\n")
                self._status("analysing image")
                next_step = templates.tool_output_prompt(
                    function_name,
                    f"Image ready and attached below. Saved at: {image_path}. It is already in front of "
                    "you — look at it directly, no further tool call needed to see it.",
                    is_last_hop,
                )
                messages.append(self.attach_image({"role": "user", "content": next_step}, image_data_uri))
            else:
                self._status(f"reading {function_name} result")
                messages.append({
                    "role": "user",
                    "content": templates.tool_output_prompt(function_name, tool_output, is_last_hop)
                })

            if is_last_hop:
                # The hop budget is spent and the last action just succeeded;
                # nothing above forces a reply, so force one now rather than
                # silently ending the turn with no visible answer.
                yield from self._final_answer_guaranteed(messages)
                return

    @staticmethod
    def _call_signature(tool_call_match: re.Match):
        """(name, canonical-args-json) for exact-repeat detection, or None if
        the call doesn't even parse (that's ToolCallError's job to report)."""
        try:
            info = json.loads(tool_call_match.group(1).strip())
            return info.get("name"), json.dumps(info.get("arguments", {}), sort_keys=True)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None

    def _final_answer_guaranteed(self, messages: list):
        """Forces one last completion with an instruction that makes an
        empty reply impossible, for use when the hop budget runs out. Falls
        back to a fixed sentence if the model still produces nothing."""
        messages.append({"role": "user", "content": templates.force_final_answer()})
        text, _, _, degenerate_start = yield from self._stream_completion(messages)
        if not text.strip() or degenerate_start:
            fallback = "I wasn't able to finish that with the tools available — could you rephrase or narrow the request?"
            yield fallback

    def _unsourced_foreign(self, text: str, sourced_text: str) -> bool:
        """True when text contains foreign-script characters that did NOT come
        out of a tool this turn (i.e. the model hand-wrote them)."""
        chars = set(_FOREIGN_SCRIPT_PATTERN.findall(text))
        return any(c not in sourced_text for c in chars)

    def _stream_completion(self, messages: list, sourced_text: str = ""):
        """Streams one completion token-by-token as it's generated, using the
        model card's sampling_params (temperature etc.) as-is.

        Scans the stream for '<think>...</think>' and '<tool_call>' tags
        wherever they appear — not just as the literal first bytes of the
        response. Plain text flushes live as it arrives; only the tail that
        could still be the start of one of these tags is held back. A
        '<think>' block is fully consumed and reported via trace_cb (never
        shown as answer text); a '<tool_call>' found anywhere stops live
        yielding and marks the response as a tool call, matching what
        run_stream's regex search over the full text already does.

        Additionally guards against the model hand-writing foreign script it
        can't write reliably: the moment an unsourced foreign character shows
        up in text about to be flushed, yielding stops (the rest is consumed
        silently) and the caller is told via the third return value so it can
        redirect through translate.

        Also guards against a known small-model failure where the FIRST
        flushed text (i.e. no '<think>' block at all — a real answer almost
        always opens with one) is just a leaked chat-role word ("user").
        Since tokens are yielded live as they arrive, this can only be caught
        by holding back the very start of any un-tagged text until enough of
        it has arrived to rule the pattern out (or a newline confirms it) —
        by the time text has been yielded once, it's already on the user's
        screen and can't be un-shown. If the holdback resolves to a match,
        nothing from this call was ever yielded; the caller can safely retry.

        Returns (full_text, is_tool_call, unsourced_foreign, degenerate_start)
        via StopIteration.value, for callers driving this with `yield from`.
        """
        pending = ""
        in_think = False
        tool_call_found = False
        suppressed = False
        degenerate_start = False
        flushed_anything = False
        pending_start = ""
        full_text = ""

        # Longest tag-open string either scanner needs to watch for as a
        # possible partial match at the tail of `pending`.
        tag_opens = (_THINK_OPEN, _TOOL_CALL_OPEN)
        # Long enough to be past any possible role-word + delimiter, short
        # enough that a real answer's first flush is barely delayed.
        _DEGENERATE_HOLDBACK_CHARS = len("assistant") + 2

        def release_pending_start():
            """Committed: pending_start is real content, not a degenerate
            prefix. Marks it flushed and returns the text to yield."""
            nonlocal flushed_anything
            flushed_anything = True
            return pending_start

        def flush(text: str):
            nonlocal suppressed, degenerate_start, pending_start
            if not text or suppressed:
                return
            if not flushed_anything:
                # Still deciding whether this is the start of a degenerate
                # reply (only relevant for the FIRST text ever flushed - once
                # real content has been shown, later text is trusted as-is).
                # Hold back until either a line break settles it or we've
                # seen enough characters to be safely past any role-word.
                # A leading newline right after </think> is normal protocol
                # formatting (see templates.py's own examples) and carries no
                # signal by itself - require at least one non-whitespace
                # character before a newline can settle the check, otherwise
                # "\n" alone (which strips to "") would look indistinguishable
                # from a truly empty reply and get misflagged as degenerate.
                pending_start += text
                has_content = bool(pending_start.strip())
                if (not has_content or "\n" not in pending_start) and len(pending_start) < _DEGENERATE_HOLDBACK_CHARS:
                    return
                if _is_degenerate_reply(pending_start) or _DEGENERATE_PREFIX_PATTERN.match(pending_start):
                    suppressed = True
                    degenerate_start = True
                    return
                text = release_pending_start()
            if self._unsourced_foreign(text, sourced_text):
                suppressed = True
                return
            yield text

        stream = self.provider.stream_completion(messages, **self.sampling_params)
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content")
            if not delta:
                continue
            full_text += delta

            if suppressed or tool_call_found:
                continue

            pending += delta

            while True:
                if in_think:
                    close_idx = pending.find(_THINK_CLOSE)
                    if close_idx == -1:
                        break
                    think_text = pending[:close_idx].strip()
                    self._trace("thought", text=think_text)
                    pending = pending[close_idx + len(_THINK_CLOSE):]
                    in_think = False
                    continue

                think_idx = pending.find(_THINK_OPEN)
                call_idx = pending.find(_TOOL_CALL_OPEN)
                candidates = [i for i in (think_idx, call_idx) if i != -1]
                if not candidates:
                    # No tag found yet - only hold back a tail that could
                    # still become the start of one; flush the rest live.
                    safe_len = len(pending)
                    for opener in tag_opens:
                        for k in range(min(len(opener), len(pending)), 0, -1):
                            if pending[-k:] == opener[:k]:
                                safe_len = min(safe_len, len(pending) - k)
                                break
                    if safe_len > 0:
                        yield from flush(pending[:safe_len])
                        pending = pending[safe_len:]
                    break

                first_idx = min(candidates)
                yield from flush(pending[:first_idx])
                if suppressed:
                    break

                if first_idx == think_idx:
                    pending = pending[first_idx + len(_THINK_OPEN):]
                    in_think = True
                    continue
                else:
                    tool_call_found = True
                    pending = ""
                    break

        if not tool_call_found and not suppressed and pending:
            # Stream ended with leftover text that never resolved into a
            # complete tag (e.g. an unclosed '<think>' or a false-alarm
            # partial match) - it was never actually a tag, so flush it as
            # literal text.
            yield from flush(pending)

        if not tool_call_found and not suppressed and not flushed_anything and pending_start:
            # Stream ended (EOS) while still holding back the very start of
            # the reply for the degenerate-prefix check — e.g. a bare "user"
            # with no trailing newline, or a short real answer ("Yes.") that
            # never reached the holdback threshold. Resolve it now against
            # the full pending_start rather than a partial prefix.
            if _is_degenerate_reply(pending_start):
                suppressed = True
                degenerate_start = True
            else:
                yield release_pending_start()

        full_text = _THINK_PATTERN.sub("", full_text).strip()
        return full_text, tool_call_found, suppressed, degenerate_start

    def _execute_tool_call(self, tool_call_match: re.Match) -> tuple[str, str]:
        """Parses and runs one ReAct action, returning (tool_output, function_name).
        Raises ToolCallError on any failure so the loop can feed it back as an
        error observation."""
        try:
            tool_info = json.loads(tool_call_match.group(1).strip())
        except (json.JSONDecodeError, AttributeError) as e:
            raise ToolCallError(f"tool call is not valid JSON ({e})")

        function_name = tool_info.get("name")
        function_args = tool_info.get("arguments", {}) or {}
        thought = str(tool_info.get("thought", "")).strip()

        if not function_name or function_name in ("tool_name", "exact_tool_name"):
            raise ToolCallError("no real tool name given — use an exact name from the TOOLS list")

        if function_name not in registry.functions:
            raise ToolCallError(
                f"tool '{function_name}' does not exist. Available: {list(registry.functions.keys())}",
                tool_name=function_name,
            )

        missing = [arg for arg in registry.required_args(function_name) if arg not in function_args]
        if missing:
            raise ToolCallError(
                f"missing required argument(s) {missing} for tool '{function_name}'",
                tool_name=function_name,
            )

        self._status(thought or f"using {function_name}")
        self._trace("tool_call", name=function_name, arguments=function_args, thought=thought)

        import inspect
        func = registry.functions[function_name]
        sig = inspect.signature(func)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if not has_kwargs:
            sanitized_args = {
                k: v for k, v in function_args.items()
                if k in sig.parameters
            }
        else:
            sanitized_args = function_args

        try:
            tool_output = func(**sanitized_args)
        except TypeError as e:
            raise ToolCallError(f"bad arguments for '{function_name}': {e}", tool_name=function_name)

        if tool_output.startswith(IMAGE_SENTINEL):
            image_path = tool_output[len(IMAGE_SENTINEL):].partition("\n")[0]
            shown = f"(image attached, saved at {image_path})"
        else:
            shown = tool_output
        self._trace("tool_result", name=function_name, result=shown)

        return tool_output, function_name
