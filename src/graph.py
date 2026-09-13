
import json
import operator
import os
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.models import call_model
from src.permissions import run_with_permission
from src.tools import MUTATING_TOOLS, TOOL_FUNCTIONS, TOOL_SCHEMAS, run_tests

MAX_FIX_RETRIES = 4

# --- Phase 5: read-only research subagents ---

READ_ONLY_SUBAGENT_TOOLS = {"list_files", "lookup_symbol", "grep", "read_file"}
RESEARCH_TOOL_SCHEMAS = [s for s in TOOL_SCHEMAS if s["function"]["name"] in READ_ONLY_SUBAGENT_TOOLS]
MAX_SUBAGENT_TOOL_ROUNDS = 7
MAX_RESEARCH_SCOPES = 3

EXCLUDED_SCOPE_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"}


class ResearchState(TypedDict):
    question: str
    findings: Annotated[list[str], operator.add]
    summary: str


def _pick_research_scopes(root: str = ".", max_scopes: int = MAX_RESEARCH_SCOPES) -> list[str]:
    """Split the repo into up to `max_scopes` independent areas to research
    in parallel — top-level directories if there are enough, else files."""
    try:
        dirs = sorted(
            e
            for e in os.listdir(root)
            if os.path.isdir(os.path.join(root, e))
            and not e.startswith(".")
            and e not in EXCLUDED_SCOPE_DIRS
        )
    except OSError:
        dirs = []

    if len(dirs) >= 2:
        return dirs[:max_scopes]

    files = sorted(f for f in os.listdir(root) if f.endswith(".py"))
    return files[:max_scopes] or ["."]


def _run_research_subagent(scope: str, question: str) -> str:
    """A small, self-contained ReAct loop scoped to read-only tools, run
    inside one Send() branch. Bounded by MAX_SUBAGENT_TOOL_ROUNDS so a
    confused subagent can't loop forever."""
    system_prompt = (
        f"You are a read-only research subagent. Focus only on the '{scope}' part of "
        "this codebase. You may use list_files, grep, lookup_symbol, and read_file — you cannot "
        "edit files or run shell commands. Answer concisely, citing specific files, "
        "how this part of the codebase relates to the question below. If it doesn't "
        "relate, say so briefly."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_SUBAGENT_TOOL_ROUNDS):
        try:
            message = call_model(messages, tools=RESEARCH_TOOL_SCHEMAS)
        except Exception as exc:
            # Groq rejects the whole request (not just the call) if the model
            # tries a tool outside the ones we sent it — correct it and retry
            # rather than crashing this branch.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {exc}. You may only use these tools: "
                        f"{', '.join(sorted(READ_ONLY_SUBAGENT_TOOLS))}."
                    ),
                }
            )
            continue

        assistant_message = {"role": "assistant", "content": message.content or ""}

        if not message.tool_calls:
            messages.append(assistant_message)
            break

        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
        messages.append(assistant_message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            if name not in READ_ONLY_SUBAGENT_TOOLS:
                result = f"[denied] research subagents may not call '{name}'"
            else:
                try:
                    result = TOOL_FUNCTIONS[name](**args)
                except Exception as exc:
                    result = f"[error running {name}]: {exc}"
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "name": name, "content": str(result)}
            )
    else:
        messages.append(
            {"role": "assistant", "content": "[subagent stopped: max tool-call rounds reached]"}
        )

    final_text = messages[-1]["content"] or "(no answer)"
    return f"### {scope}\n{final_text}"


def research_worker_node(state: dict) -> dict:
    start = time.time()
    print(f"[research:{state['scope']}] started")
    finding = _run_research_subagent(state["scope"], state["question"])
    print(f"[research:{state['scope']}] finished in {time.time() - start:.2f}s")
    return {"findings": [finding]}


def fan_out_research(state: ResearchState):
    scopes = _pick_research_scopes(".")
    return [Send("research_worker", {"scope": s, "question": state["question"]}) for s in scopes]


def synthesize_node(state: ResearchState) -> dict:
    combined = "\n\n".join(state["findings"])
    message = call_model(
        [
            {
                "role": "system",
                "content": (
                    "Synthesize these independent research findings into one clear summary "
                    "that answers the original question. Note where the findings connect; "
                    "don't let any single one dominate."
                ),
            },
            {"role": "user", "content": f"Question: {state['question']}\n\nFindings:\n{combined}"},
        ]
    )
    return {"summary": message.content or combined}


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("research_worker", research_worker_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_conditional_edges(START, fan_out_research, ["research_worker"])
    graph.add_edge("research_worker", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


_research_graph = None


def research_codebase(question: str) -> str:
    """Fan out 2-3 independent read-only subagents across different areas
    of the codebase to research a broad question, then synthesize their
    findings into one summary."""
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    result = _research_graph.invoke({"question": question, "findings": []})
    return result["summary"]


RESEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "research_codebase",
        "description": (
            "Answer a broad, cross-cutting question about how different parts of the "
            "codebase connect (e.g. 'how does X relate to Y across this project'), by "
            "fanning out independent read-only subagents across different areas and "
            "synthesizing their findings. Use lookup_symbol instead for a single known "
            "symbol's location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The broad question to research."},
            },
            "required": ["question"],
        },
    },
}

MAIN_TOOL_SCHEMAS = TOOL_SCHEMAS + [RESEARCH_TOOL_SCHEMA]
MAIN_TOOL_FUNCTIONS = {**TOOL_FUNCTIONS, "research_codebase": research_codebase}


# --- Main agent loop ---


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    retry_count: int
    last_signature: str | None
    next: str


def agent_node(state: AgentState) -> dict:
    message = call_model(state["messages"], tools=MAIN_TOOL_SCHEMAS)

    assistant_message = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]

    return {"messages": [assistant_message]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"
    return END


def tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_results = []
    edit_applied = False
    call_signatures = []

    for tool_call in last_message["tool_calls"]:
        name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        args = json.loads(arguments)
        func = MAIN_TOOL_FUNCTIONS[name]

        try:
            result = run_with_permission(name, args, func, MUTATING_TOOLS)
        except Exception as exc:
            result = f"[error running {name}]: {exc}"

        if name == "edit_file" and str(result).startswith("[edit applied]"):
            edit_applied = True

        tool_results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": name,
                "content": str(result),
            }
        )
        # tool_call_id is unique per call even for a genuinely repeated
        # call, so the repetition signature is keyed on (name, arguments,
        # result) instead — the actual content of the round, not its id.
        call_signatures.append((name, arguments, str(result)))

    signature = json.dumps(call_signatures, sort_keys=True)

    if signature == state.get("last_signature"):
        tool_results.append(
            {
                "role": "assistant",
                "content": (
                    "[repetition guard] the last tool call and its result are identical to the "
                    "previous round. Stopping here instead of retrying forever."
                ),
            }
        )
        return {"messages": tool_results, "next": END}

    if edit_applied:
        return {"messages": tool_results, "last_signature": signature, "next": "run_tests"}

    return {"messages": tool_results, "last_signature": signature, "next": "agent"}


def route_after_tools(state: AgentState) -> str:
    return state["next"]


def run_tests_node(state: AgentState) -> dict:
    output = run_tests(".")
    passed = output.startswith("[tests PASSED]")

    if passed:
        message = {"role": "user", "content": f"Automated test run after your edit:\n{output}"}
        return {"messages": [message], "retry_count": 0, "next": "agent"}

    retry_count = state.get("retry_count", 0) + 1
    if retry_count > MAX_FIX_RETRIES:
        message = {
            "role": "user",
            "content": (
                f"Automated test run after your edit failed again "
                f"(attempt {retry_count}/{MAX_FIX_RETRIES}):\n{output}"
            ),
        }
        stop_message = {
            "role": "assistant",
            "content": (
                f"[stopped] tests are still failing after {MAX_FIX_RETRIES} fix attempts. "
                "Stopping instead of retrying indefinitely."
            ),
        }
        return {"messages": [message, stop_message], "retry_count": retry_count, "next": END}

    message = {
        "role": "user",
        "content": (
            f"Automated test run after your edit failed "
            f"(attempt {retry_count}/{MAX_FIX_RETRIES}):\n{output}"
        ),
    }
    return {"messages": [message], "retry_count": retry_count, "next": "agent"}


def route_after_tests(state: AgentState) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("run_tests", run_tests_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_conditional_edges(
        "tools", route_after_tools, {"agent": "agent", "run_tests": "run_tests", END: END}
    )
    graph.add_conditional_edges("run_tests", route_after_tests, {"agent": "agent", END: END})

    return graph.compile()
