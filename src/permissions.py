"""Permission gate for mutating tools.

Read-only tools run silently. Mutating tools print the pending action and
wait for y/n approval, auto-denying if the user doesn't respond within
TIMEOUT_SECONDS (lavalamp's "auto-deny on timeout" behavior).

The actual I/O (how the user is asked, and how) is swappable via
set_ask_backend — the approval logic in request_approval (what counts as
approved, timeout means deny) never changes regardless of backend. The
default backend is a plain terminal prompt; src/tui.py swaps in a modal
dialog without touching anything below.
"""

import select
import sys

TIMEOUT_SECONDS = 30


def _prompt_with_timeout(prompt: str, timeout: int) -> str | None:
    """Print `prompt` and wait up to `timeout` seconds for a line of input.
    Returns the input (stripped) or None if the timeout elapses first."""
    print(prompt, end="", flush=True)
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.readline().strip()
    return None


def _default_ask_backend(tool_name: str, args: dict, timeout: int) -> str | None:
    """Terminal-based default backend: print the pending action, wait for a
    y/n line, print the outcome. Returns the raw typed answer, or None on
    timeout — the contract every backend must follow."""
    print(f"\n[permission] about to run '{tool_name}' with args: {args}")
    answer = _prompt_with_timeout(f"Allow this? [y/n] (auto-deny in {timeout}s): ", timeout)
    if answer is None:
        print("\n[permission] timed out — auto-denied.")
    elif answer.lower() not in ("y", "yes"):
        print("[permission] denied.")
    return answer


_ask_backend = _default_ask_backend


def set_ask_backend(fn) -> None:
    """Swap how the user is asked for approval. `fn(tool_name, args,
    timeout) -> str | None` must return the raw answer (or None on
    timeout) — same contract as _default_ask_backend."""
    global _ask_backend
    _ask_backend = fn


def request_approval(tool_name: str, args: dict) -> bool:
    """Ask the user to approve a mutating tool call. Returns True if approved."""
    answer = _ask_backend(tool_name, args, TIMEOUT_SECONDS)
    if answer is None:
        return False
    return answer.lower() in ("y", "yes")


def run_with_permission(tool_name: str, args: dict, func, mutating_tools: set) -> str:
    """Run `func(**args)` directly if `tool_name` is not mutating. Otherwise,
    gate it behind request_approval() first, returning a denial message
    instead of running it if the user denies or times out."""
    if tool_name not in mutating_tools:
        return func(**args)

    if not request_approval(tool_name, args):
        return f"[denied by user] tool '{tool_name}' was not executed."

    return func(**args)
