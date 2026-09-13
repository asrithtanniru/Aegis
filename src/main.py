"""CLI entrypoint. Plain input()/print() chat loop — no TUI, per plan."""

import os
import sys

from src.graph import build_graph

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


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(root)

    graph = build_graph()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Aegis ready. Working directory: {os.getcwd()}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        result = graph.invoke(
            {"messages": messages, "retry_count": 0, "last_signature": None}
        )
        messages = result["messages"]

        final_message = messages[-1]
        print(final_message["content"])


if __name__ == "__main__":
    main()
