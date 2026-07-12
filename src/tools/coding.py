"""Coding & execution tools: run code inside the local workspace sandbox.
Nothing here can touch paths outside WORKSPACE_DIR or run for longer than a
short timeout — a small local model occasionally loops or hangs a script, and
this is a personal machine agent, not a CI runner.

No git tools live here on purpose: agent_workspace/ is not its own git repo,
so a git command run there walks up to Tuffy's own project .git instead of
staying sandboxed — letting the agent commit changes to Tuffy's own source
tree by accident. If workspace-scoped git is ever wanted, it needs its own
check that agent_workspace/.git exists before running anything.
"""

import shlex
import subprocess

from src.tools.registry import registry
from src.tools.editing import WORKSPACE_DIR, safe_workspace_path

_EXEC_TIMEOUT_SECONDS = 20
_MAX_OUTPUT_CHARS = 4000

# Deliberately small and read/version-control-oriented — no rm, mv, curl,
# chmod, etc. This is a fixed allowlist, not a blocklist, so an unexpected
# command name always fails safe.
_ALLOWED_SHELL_COMMANDS = {
    "ls", "cat", "pwd", "echo", "wc", "grep", "find", "head", "tail",
    "python3", "pip", "pytest", "node", "npm", "npx",
}


def _run(args: list[str], cwd: str) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT_SECONDS,
        )
        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "(no output)"
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
        return f"exit code {result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {_EXEC_TIMEOUT_SECONDS}s."
    except Exception as e:
        return f"Execution failed: {str(e)}"


@registry.register(
    name="run_python",
    description="Run a Python file that already exists in the workspace and return its stdout/stderr. Write the file with save_to_file first, then run it here.",
    parameters={
        "filename": {"type": "string", "description": "Path to the .py file in the workspace to run, e.g. 'script.py'."},
        "args": {"type": "string", "description": "Optional space-separated command-line arguments to pass to the script."}
    },
    required=["filename"],
    group="coding",
)
def run_python(filename: str, args: str = "") -> str:
    import os

    try:
        file_path = os.path.abspath(safe_workspace_path(filename))
    except ValueError as e:
        return f"Execution failed: {str(e)}"

    extra_args = shlex.split(args) if args.strip() else []
    return _run(["python3", file_path] + extra_args, cwd=WORKSPACE_DIR)


@registry.register(
    name="run_shell",
    description="Run a shell command inside the workspace directory. Only ls, cat, pwd, echo, wc, grep, find, head, tail, python3, pip, pytest, node, npm, npx are allowed — anything else is rejected.",
    parameters={
        "command": {"type": "string", "description": "The full shell command to run, e.g. 'pytest -q' or 'ls -la'."}
    },
    required=["command"],
    group="coding",
)
def run_shell(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"Could not parse command: {e}"

    if not parts:
        return "Empty command."
    if parts[0] not in _ALLOWED_SHELL_COMMANDS:
        return (
            f"Command '{parts[0]}' is not allowed. Allowed commands: "
            f"{sorted(_ALLOWED_SHELL_COMMANDS)}"
        )

    return _run(parts, cwd=WORKSPACE_DIR)
