"""Deterministic tests for Phase 4's fix-loop retry cap and repetition
guard, using a fake model instead of a real LLM call.

A real model can be too clever to reliably get stuck (see AEGIS_PLAN.md's
Phase 4 build log — Qwen found a legitimate workaround instead of looping),
so these tests drive the graph's own control flow directly to prove the
guard logic fires when a model *does* get stuck.
"""

from types import SimpleNamespace

import src.graph as graph_module
from src.graph import MAX_FIX_RETRIES, build_graph


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_repetition_guard_stops_on_identical_repeat(monkeypatch):
    """A model that issues the exact same tool call twice in a row (and
    gets the exact same result back) should be stopped after the second
    round, not retried forever."""
    call_count = {"n": 0}

    def fake_call_model(messages, tools=None, model=None):
        call_count["n"] += 1
        return SimpleNamespace(
            content="",
            tool_calls=[_tool_call(f"c{call_count['n']}", "noop", '{"x": 1}')],
        )

    monkeypatch.setattr(graph_module, "call_model", fake_call_model)
    monkeypatch.setitem(graph_module.MAIN_TOOL_FUNCTIONS, "noop", lambda x: "same result")
    monkeypatch.setattr(graph_module, "MUTATING_TOOLS", set())  # no permission prompt

    app = build_graph()
    result = app.invoke(
        {
            "messages": [{"role": "user", "content": "do the thing"}],
            "retry_count": 0,
            "last_signature": None,
        }
    )

    assert call_count["n"] == 2, "should stop after the second identical round, not keep going"
    assert any("[repetition guard]" in (m.get("content") or "") for m in result["messages"])


def test_retry_cap_stops_after_max_attempts(monkeypatch):
    """A model that keeps making distinct edits that never fix the tests
    should be stopped after MAX_FIX_RETRIES attempts, not loop forever."""
    call_count = {"n": 0}

    def fake_call_model(messages, tools=None, model=None):
        call_count["n"] += 1
        # distinct args each round so the repetition guard doesn't trip first
        args = f'{{"path": "fake.py", "old_text": "x{call_count["n"]}", "new_text": "y{call_count["n"]}"}}'
        return SimpleNamespace(
            content="",
            tool_calls=[_tool_call(f"c{call_count['n']}", "edit_file", args)],
        )

    monkeypatch.setattr(graph_module, "call_model", fake_call_model)
    monkeypatch.setitem(graph_module.MAIN_TOOL_FUNCTIONS, "edit_file", lambda **kw: "[edit applied] fake.py updated.")
    monkeypatch.setattr(graph_module, "MUTATING_TOOLS", set())  # no permission prompt
    monkeypatch.setattr(graph_module, "run_tests", lambda path: "[tests FAILED]\nstill broken")

    app = build_graph()
    result = app.invoke(
        {
            "messages": [{"role": "user", "content": "fix the tests"}],
            "retry_count": 0,
            "last_signature": None,
        }
    )

    assert call_count["n"] == MAX_FIX_RETRIES + 1, "should stop right after exceeding the retry cap"
    assert any("[stopped]" in (m.get("content") or "") for m in result["messages"])
