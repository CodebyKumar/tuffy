"""The CLI: everything that turns Tuffy's agent core into an interactive
terminal chat. Split by responsibility so main.py stays a thin entry point:

  display.py   - colors, the startup banner, the animated status spinner
  session.py   - Session (active model/agent, chat history, history trimming)
  commands.py  - one function per slash command + the dispatch table
  turn.py      - runs a single user turn: stream tokens, update history

Nothing outside this package prints to the terminal or reads stdin — that
keeps src/agent.py and the tool/model/skill registries usable headless
(e.g. from a future non-CLI frontend) without change.
"""
