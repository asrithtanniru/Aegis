"""Permission gate for mutating tools.

Read-only tools run silently. Mutating tools print the pending action and
wait for y/n approval, auto-denying if the user doesn't respond within
TIMEOUT_SECONDS (lavalamp's "auto-deny on timeout" behavior).
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


def request_approval(tool_name: str, args: dict) -> bool:
    """Ask the user to approve a mutating tool call. Returns True if approved."""
    print(f"\n[permission] about to run '{tool_name}' with args: {args}")
    answer = _prompt_with_timeout(
        f"Allow this? [y/n] (auto-deny in {TIMEOUT_SECONDS}s): ", TIMEOUT_SECONDS
    )
    if answer is None:
        print("\n[permission] timed out — auto-denied.")
        return False
    approved = answer.lower() in ("y", "yes")
    if not approved:
        print("[permission] denied.")
    return approved


def run_with_permission(tool_name: str, args: dict, func, mutating_tools: set) -> str:
    """Run `func(**args)` directly if `tool_name` is not mutating. Otherwise,
    gate it behind request_approval() first, returning a denial message
    instead of running it if the user denies or times out."""
    if tool_name not in mutating_tools:
        return func(**args)

    if not request_approval(tool_name, args):
        return f"[denied by user] tool '{tool_name}' was not executed."

    return func(**args)
