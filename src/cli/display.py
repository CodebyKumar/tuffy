"""Terminal presentation: ANSI colors, the startup banner, and the animated
status spinner. Nothing in here touches Session state or agent internals —
it only renders strings it's handed."""

import sys
import threading
import time

C_AI = "\033[38;2;0;255;180m"   # Mint
C_USER = "\033[96m"             # Cyan
C_DIM = "\033[2m"               # Faded gray
C_SUCCESS = "\033[92m"          # Green
C_WARN = "\033[93m"             # Yellow
C_BLUE = "\033[94m"             # Blue
C_BOLD = "\033[1m"
C_RESET = "\033[0m"
CLEAR_LINE = "\r\033[K"

_BANNER = f"""{C_SUCCESS}{C_BOLD}
  ████████╗██╗   ██╗███████╗███████╗██╗   ██╗
  ╚══██╔══╝██║   ██║██╔════╝██╔════╝╚██╗ ██╔╝
     ██║   ██║   ██║█████╗  █████╗   ╚████╔╝
     ██║   ██║   ██║██╔══╝  ██╔══╝    ╚██╔╝
     ██║   ╚██████╔╝██║     ██║        ██║
     ╚═╝    ╚═════╝ ╚═╝     ╚═╝        ╚═╝
     {C_RESET}"""


def print_logo():
    """Just the ASCII art — no model/status text baked into it, so startup
    info (which model loaded, whether vision is on) always reads as
    something that happened *after* the banner, not part of it."""
    print(_BANNER)


def print_session_info(model_name: str, vision: bool):
    """Printed once, after the logo and after the model has finished
    loading — a one-line summary of what's active, then a pointer to /help
    (which has the full command list, so this line doesn't need to)."""
    tag = "vision" if vision else "text-only"
    print(f"{C_DIM}Active model: {model_name} ({tag}){C_RESET}")
    print(f"{C_DIM}Type /help to see everything Tuffy can do.{C_RESET}\n")


class Spinner:
    """Terminal spinner for AI status updates."""

    MAX_LABEL = 64

    def __init__(self, label: str = "thinking"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def set_label(self, label: str):
        label = " ".join(str(label).split())

        if len(label) > self.MAX_LABEL:
            label = label[: self.MAX_LABEL - 1] + "…"

        with self._lock:
            self.label = label or "thinking"

    def start(self):
        if self._thread is not None:
            return

        self._stop_event.clear()

        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

        def run():
            frames = ["", ".", "..", "..."]

            i = 0
            while not self._stop_event.is_set():
                with self._lock:
                    label = self.label

                print(
                    f"{CLEAR_LINE}{C_AI}AI ❯{C_RESET} "
                    f"{C_DIM}{label}{frames[i % len(frames)]}{C_RESET}",
                    end="",
                    flush=True,
                )

                i += 1
                time.sleep(0.4)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join()

        self._thread = None

        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

        print(
            f"{CLEAR_LINE}{C_AI}AI ❯{C_RESET} ",
            end="",
            flush=True,
        )
