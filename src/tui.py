"""Textual TUI for Aegis.

UI layer only — replaces main.py's plain input()/print() loop. graph.py's
loop logic, tools.py, and indexer.py are untouched. permissions.py's
approval LOGIC (what counts as approved, timeout means deny) is also
untouched — only the I/O mechanism is swapped, via its set_ask_backend hook.

The conversation transcript is a VerticalScroll ("#chat-log") that every
message — user echo, tool status, permission prompts, assistant text —
mounts into as its own widget, auto-scrolling to the end each time. This
replaced an earlier RichLog-based design: mixing a RichLog with separately
mounted widgets squeezed everything above the input box instead of flowing
naturally as a growing, scrolling transcript.
"""

import json
import os
import re
import sys
import threading
import time

from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, LoadingIndicator, Static

from src.graph import build_graph
from src.models import DEFAULT_MODEL
from src.permissions import set_ask_backend
from src.splash import build_splash

SYSTEM_PROMPT = (
    "You are Aegis, a coding agent. You can read files, list files, look up "
    "where a function or class is defined, edit files, and run shell "
    "commands in the current project directory. "
    "For any file modification, use edit_file with an exact old_text/new_text "
    "pair rather than run_shell or rewriting the whole file. To find where a "
    "symbol is defined, use lookup_symbol instead of reading every file. "
    "After you edit a file, tests will run automatically and any failure "
    "will be reported back to you so you can try again. "
    "For a broad, cross-cutting question about how different parts of the "
    "codebase connect, use research_codebase instead of manually exploring "
    "each part yourself. "
    "Use tools when you need information or need to take action. Respond in "
    "plain text when you have a final answer and no further tool calls are "
    "needed."
)

AGENT_STYLE = "#d29922"
TOOL_NAME_STYLE = "#58a6ff"
TOOL_STYLE = "#8b949e"
PASS_STYLE = "#3fb950"
FAIL_STYLE = "#f85149"
DIFF_REMOVED_BG = "#3d1a1a"
DIFF_ADDED_BG = "#1a3d1a"
BRANCH_PALETTE = ["#bc8cff", "#ffa657", "#39c5cf"]

OPTIONS = [("y", "Allow"), ("n", "Deny")]

_RESEARCH_LINE_RE = re.compile(r"^\[research:(.+?)\] (started|finished.*)$")


class _StreamingStdout:
    """Redirects process-wide stdout for the duration of a turn, so
    run_shell's existing per-line print() calls (tools.py, unmodified)
    reach the chat log as they happen instead of being lost — Textual owns
    the real terminal now, plain print() has nowhere else to go. Buffers
    until a newline so partial writes don't produce broken lines.

    research_codebase's subagent branches (graph.py, unmodified) already
    print "[research:{scope}] started/finished" — parsed here and colored
    per scope (round-robin across BRANCH_PALETTE) so concurrent branches
    read as visually distinct, without touching graph.py at all.
    """

    def __init__(self, app: "AegisApp"):
        self.app = app
        self._buffer = ""
        self._branch_colors: dict[str, str] = {}

    def write(self, text: str) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._dispatch(line)

    def _dispatch(self, line: str) -> None:
        # write() can in principle be called from the main thread too (any
        # print() anywhere while stdout is redirected) — call_from_thread
        # raises if not actually called cross-thread, so fall back to a
        # direct call in that case instead of crashing.
        try:
            self.app.call_from_thread(self._write_line, line)
        except RuntimeError:
            self._write_line(line)

    def _color_for_branch(self, scope: str) -> str:
        if scope not in self._branch_colors:
            index = len(self._branch_colors) % len(BRANCH_PALETTE)
            self._branch_colors[scope] = BRANCH_PALETTE[index]
        return self._branch_colors[scope]

    def _write_line(self, line: str) -> None:
        match = _RESEARCH_LINE_RE.match(line)
        style = self._color_for_branch(match.group(1)) if match else TOOL_STYLE
        self.app.append_message(Text(line, style=style))

    def flush(self) -> None:
        if self._buffer:
            self._dispatch(self._buffer)
            self._buffer = ""


def _guess_lexer(path: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".json": "json",
        ".md": "markdown", ".sh": "bash", ".css": "css", ".html": "html",
        ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    }
    _, ext = os.path.splitext(path)
    return ext_map.get(ext, "text")


def _render_edit_diff(args: dict):
    """A syntax-highlighted diff view for an edit_file call: old_text on a
    dark-red-tinted block, new_text on a dark-green-tinted block — same
    idea as a unified diff, just shown as two whole blocks rather than a
    line-by-line match, since edit_file's old_text/new_text are already
    exactly the two sides of the patch."""
    path = args.get("path", "")
    old_text = args.get("old_text", "") or "(empty)"
    new_text = args.get("new_text", "") or "(empty)"
    lexer = _guess_lexer(path)

    return Group(
        Text(path, style=TOOL_STYLE),
        Syntax(old_text, lexer, background_color=DIFF_REMOVED_BG, word_wrap=True),
        Syntax(new_text, lexer, background_color=DIFF_ADDED_BG, word_wrap=True),
    )


class PermissionPrompt(Widget):
    """Inline approval prompt, mounted directly in the conversation flow
    (not a popup). Arrow keys + Enter to choose, or y/n as a shortcut.
    Pure UI — carries no approval logic of its own; that stays in
    permissions.py. Calls on_result(bool) once, then removes itself."""

    can_focus = True

    def __init__(self, tool_name: str, args: dict, timeout: int, on_result):
        super().__init__(id="permission-prompt")
        self.tool_name = tool_name
        self.args = args
        self.timeout = timeout
        self.remaining = timeout
        self.selected = 0
        self.on_result = on_result
        self._done = False

    def compose(self) -> ComposeResult:
        yield Static("Permission required", id="permission-title")
        yield Static(self._render_detail(), id="permission-detail")
        yield Static(self._options_text(), id="permission-options")
        yield Static(f"auto-deny in {self.remaining}s", id="permission-countdown")

    def _render_detail(self):
        if self.tool_name == "edit_file":
            return _render_edit_diff(self.args)
        if self.tool_name == "run_shell":
            command = self.args.get("command", "")
            return Group(Text("Command:", style=TOOL_STYLE), Text(command))
        return Group(
            Text(f"Tool: {self.tool_name}", style=TOOL_STYLE),
            Text(str(self.args)),
        )

    def _options_text(self) -> Text:
        text = Text()
        for i, (key, label) in enumerate(OPTIONS):
            cursor = "❯ " if i == self.selected else "  "
            style = AGENT_STYLE if i == self.selected else TOOL_STYLE
            text.append(f"{cursor}[{key}] {label}", style=style)
            if i < len(OPTIONS) - 1:
                text.append("   ")
        return text

    def on_mount(self) -> None:
        self.focus()
        self._timer = self.set_interval(1, self._tick)

    def _tick(self) -> None:
        self.remaining -= 1
        self.query_one("#permission-countdown", Static).update(f"auto-deny in {self.remaining}s")
        if self.remaining <= 0:
            self._finish(False)

    def _finish(self, approved: bool) -> None:
        if self._done:
            return
        self._done = True
        self._timer.stop()
        outcome = Text("allowed", style=AGENT_STYLE) if approved else Text("denied", style=FAIL_STYLE)
        self.app.append_message(outcome)
        self.on_result(approved)
        self.remove()
        self.app.query_one("#user-input", Input).focus()

    def on_key(self, event) -> None:
        if event.key in ("up", "down", "left", "right"):
            self.selected = 1 - self.selected
            self.query_one("#permission-options", Static).update(self._options_text())
        elif event.key == "enter":
            self._finish(self.selected == 0)
        elif event.key == "y":
            self._finish(True)
        elif event.key == "n":
            self._finish(False)
        elif event.key == "1":
            self._finish(True)
        elif event.key == "2":
            self._finish(False)


def _render_tool_call_line(name: str, count: int, args_list: list) -> Text:
    """A single always-visible line: "> name | {args}" — tool name in blue,
    args in grey. No icon, no expand/collapse — the full call is always
    shown inline, not hidden behind an interaction."""
    count_str = f" ×{count}" if count > 1 else ""
    text = Text()
    text.append(f"> {name}{count_str}", style=TOOL_NAME_STYLE)
    text.append(" | ", style=TOOL_STYLE)
    text.append(", ".join(str(a) for a in args_list), style=TOOL_STYLE)
    return text


class AegisApp(App):
    CSS_PATH = "tui.css"
    # Textual 8.x rebinds ctrl+c to a "press ctrl+q to quit" notice instead
    # of actually quitting (it repurposes ctrl+c for copy in inputs). That's
    # surprising for a CLI tool — override it back to a real quit.
    BINDINGS = [Binding("ctrl+c", "quit", "Quit", priority=True)]

    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.graph = build_graph()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        set_ask_backend(self._ask_inline)

    # --- transcript helpers -------------------------------------------------

    def append_message(self, renderable) -> Static:
        """Mount a new widget at the end of the scrolling transcript and
        keep the view pinned to the bottom. The one place every message
        (text, tool status, prompts) enters the conversation."""
        chat_log = self.query_one("#chat-log", VerticalScroll)
        static = Static(renderable, markup=False, classes="message")
        chat_log.mount(static)
        chat_log.scroll_end(animate=False)
        return static

    def show_loader(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(LoadingIndicator(id="loading"))
        chat_log.scroll_end(animate=False)

    def hide_loader(self) -> None:
        loaders = self.query("#loading")
        if loaders:
            loaders.first().remove()

    # --- permission bridge ---------------------------------------------------

    def _ask_inline(self, tool_name: str, args: dict, timeout: int) -> str | None:
        """permissions.py's ask-backend contract: return the raw answer, or
        None on timeout. Called from the worker thread running graph.invoke
        — mounting the prompt must happen on the main thread
        (call_from_thread), then we block this thread on a plain
        threading.Event until the prompt is answered."""
        result: dict = {}
        done = threading.Event()

        def handle_result(approved: bool) -> None:
            result["approved"] = approved
            done.set()

        def show_prompt() -> None:
            self.hide_loader()
            chat_log = self.query_one("#chat-log", VerticalScroll)
            prompt = PermissionPrompt(tool_name, args, timeout, handle_result)
            chat_log.mount(prompt)
            chat_log.scroll_end(animate=False)

        self.call_from_thread(show_prompt)
        done.wait(timeout=timeout + 2)  # safety margin beyond the prompt's own countdown
        approved = result.get("approved")
        if approved is None:
            return None
        return "y" if approved else "n"

    # --- layout ---------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="main"):
            yield VerticalScroll(id="chat-log")
            # Input + status bar share one docked wrapper. Docking each of
            # them separately to "bottom" overlapped: the second dock got
            # squeezed onto the first one's last row instead of stacking
            # below it, pushing its content off-screen entirely.
            with Vertical(id="bottom-bar"):
                yield Input(placeholder="Ask Aegis...", id="user-input")
                with Horizontal(id="status-bar"):
                    yield Static(os.getcwd(), id="status-cwd")
                    yield Static(f"model: {DEFAULT_MODEL}", id="status-model")

    def on_mount(self) -> None:
        # Splash-only startup, lavalamp-style: no welcome text — the
        # footer already shows cwd/model, and the input's own placeholder
        # covers usage, so nothing else competes with the art for attention.
        chat_log = self.query_one("#chat-log", VerticalScroll)
        splash = Static(Text(build_splash(), style=AGENT_STYLE), classes="message splash")
        chat_log.mount(splash)
        self.query_one("#user-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text.lower() in ("exit", "quit"):
            self.exit()
            return

        self.append_message(f"> {text}")
        self.show_loader()
        self.messages.append({"role": "user", "content": text})
        self.run_worker(self.run_turn, thread=True, exclusive=True)

    def run_turn(self) -> None:
        # Runs on a worker thread so the main thread stays free to render
        # and handle an inline permission prompt if one comes up mid-turn.
        #
        # graph.stream() instead of graph.invoke() — same compiled graph, no
        # change to graph.py's node/edge logic — yields the accumulated
        # state after each node finishes, so tool results (run_tests) show
        # up as soon as they're available instead of only at the very end.
        #
        # self.messages already carries the full conversation history into
        # this call (needed for context), so the very first chunk yielded
        # already contains every prior turn's messages too. seen must start
        # at that existing count, not 0 — otherwise every past assistant
        # response gets replayed through _type_out again on every new turn.
        seen = len(self.messages)
        stdout = _StreamingStdout(self)
        old_stdout = sys.stdout
        sys.stdout = stdout
        try:
            for chunk in self.graph.stream(
                {"messages": self.messages, "retry_count": 0, "last_signature": None},
                stream_mode="values",
            ):
                messages = chunk["messages"]
                for message in messages[seen:]:
                    self.call_from_thread(self.hide_loader)
                    if (
                        message.get("role") == "assistant"
                        and not message.get("tool_calls")
                        and message.get("content")
                    ):
                        self._type_out(message["content"])
                    else:
                        self.call_from_thread(self._show_message, message)
                seen = len(messages)
                self.messages = messages
        finally:
            stdout.flush()
            self.call_from_thread(self.hide_loader)
            sys.stdout = old_stdout

    def _type_out(self, content: str, delay: float = 0.03) -> None:
        """Reveal an assistant text message word-by-word rather than all at
        once — a display-layer effect (the full text already arrived from
        Groq; models.py's call still isn't a streaming API call). Runs on
        the calling worker thread: each word update is dispatched to the
        main thread via call_from_thread, with the pacing delay happening
        here, not on the UI thread, so rendering stays smooth throughout."""
        holder: dict = {}

        def mount() -> None:
            static = self.append_message("")
            static.styles.color = AGENT_STYLE
            holder["widget"] = static

        self.call_from_thread(mount)
        static = holder["widget"]
        chat_log = self.query_one("#chat-log", VerticalScroll)

        def update_and_scroll(text: str) -> None:
            # One atomic main-thread call per word, not two — avoids any
            # render tearing between the text update and the scroll.
            static.update(text)
            chat_log.scroll_end(animate=False)

        words = content.split(" ")
        partial = ""
        for i, word in enumerate(words):
            partial += word + (" " if i < len(words) - 1 else "")
            self.call_from_thread(update_and_scroll, partial)
            time.sleep(delay)

    def _show_message(self, message: dict) -> None:
        role = message.get("role")

        if role == "user":
            # The turn's own user message is already echoed by
            # on_input_submitted; anything else with role="user" mid-turn
            # is the graph's own auto-injected fix-loop feedback (Phase 4),
            # already covered by the run_tests tool message below.
            return

        if role == "assistant":
            # Plain-text messages go through _type_out. Tool-calling ones
            # get a collapsed-by-default summary (lavalamp-style
            # "name ×N ▶"), expandable to the exact args.
            if message.get("tool_calls"):
                self._show_tool_calls(message["tool_calls"])
            return

        if role == "tool":
            self._show_tool_result(message)

    def _show_tool_calls(self, tool_calls: list[dict]) -> None:
        """One always-visible line per distinct tool name in this batch:
        "> name | {args}" — no icon, no expand/collapse."""
        groups: dict[str, list[dict]] = {}
        for call in tool_calls:
            groups.setdefault(call["function"]["name"], []).append(call)

        for name, calls in groups.items():
            args_list = []
            for call in calls:
                try:
                    args_list.append(json.loads(call["function"]["arguments"]))
                except (json.JSONDecodeError, TypeError):
                    args_list.append({})
            self.append_message(_render_tool_call_line(name, len(calls), args_list))

    def _show_tool_result(self, message: dict) -> None:
        name = message.get("name")
        content = message.get("content", "")

        if name == "run_tests":
            style = PASS_STYLE if content.startswith("[tests PASSED]") else FAIL_STYLE
            self.append_message(Text(content, style=style))
        elif name == "research_codebase":
            # The individual branches already streamed their own colored
            # started/finished lines via stdout; this is the synthesized
            # result once they've all merged.
            self.append_message(Text("research summary:", style=TOOL_STYLE))
            self.append_message(Text(content, style=AGENT_STYLE))
        # run_shell's own result stays silent (already streamed live via
        # stdout); other tool results (read_file, grep, ...) already had
        # their call shown by _show_tool_calls, nothing further to add.


def run(root: str) -> None:
    os.chdir(root)
    AegisApp(root).run()
