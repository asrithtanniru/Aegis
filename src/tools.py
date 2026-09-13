"""Core tools available to the agent. Each tool is a plain function plus a
JSON-schema entry used to build the Groq `tools` payload.

Tools are split by mutation: read_file / list_files are read-only and run
silently. run_shell can mutate the workspace or system, so it is wrapped by
permissions.py before the agent ever calls it directly.
"""

import ast
import fnmatch
import os
import re
import subprocess
import tempfile

from src.indexer import (
    find_class_location,
    find_classes_in_file,
    find_symbol_location,
    lookup_symbol as _lookup_symbol,
)

# Directories never worth walking into: vendored deps and VCS/cache
# metadata, not project code. Same list as indexer.py's EXCLUDED_DIRS.
EXCLUDED_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"}

# Cap on any single tool result fed back to the model, so one big
# directory listing or grep match set can't blow the context window
# (this bit Phase 5's research subagents in practice).
MAX_RESULT_CHARS = 4000


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n...[truncated, {len(text) - MAX_RESULT_CHARS} more chars]"


def lookup_symbol(name: str) -> str:
    """Return the file and line range where `name` (a function or class) is
    defined, using the tree-sitter/SQLite symbol index."""
    return _lookup_symbol(name)


def read_file(path: str) -> str:
    """Return the full contents of a file as a string."""
    with open(path, "r") as f:
        return _truncate(f.read())


def list_files(directory: str = ".", pattern: str = "*") -> list[str]:
    """List files under `directory` (recursively) matching a glob `pattern`.
    Skips vendored/VCS directories (see EXCLUDED_DIRS)."""
    matches = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.relpath(os.path.join(root, name), directory))
    matches = sorted(matches)
    if len(matches) > 300:
        matches = matches[:300] + [f"...[truncated, {len(matches) - 300} more files]"]
    return matches


def grep(pattern: str, directory: str = ".", file_pattern: str = "*.py") -> str:
    """Search file contents for a regex `pattern`, returning matching
    file:line: text lines. Read-only. Skips vendored/VCS directories."""
    matches = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for name in files:
            if not fnmatch.fnmatch(name, file_pattern):
                continue
            filepath = os.path.join(root, name)
            try:
                with open(filepath, "r", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if re.search(pattern, line):
                            rel = os.path.relpath(filepath, directory)
                            matches.append(f"{rel}:{lineno}: {line.rstrip()}")
            except OSError:
                continue
    if not matches:
        return f"[grep] no matches for '{pattern}'"
    return _truncate("\n".join(matches))


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace `old_text` with `new_text` in `path`, via exact string match.

    Fails loudly instead of guessing: if `old_text` doesn't appear exactly
    once in the file, nothing is written and the model is told to re-read
    the file and try again. This avoids silently patching the wrong spot,
    or corrupting the file if it changed since the model last read it.
    """
    with open(path, "r") as f:
        content = f.read()

    count = content.count(old_text)
    if count == 0:
        return (
            f"[edit failed] old_text not found in {path}. "
            "Re-read the file and try again with an exact match."
        )
    if count > 1:
        return (
            f"[edit failed] old_text matches {count} locations in {path}, expected exactly 1. "
            "Include more surrounding context to make old_text unique, then try again."
        )

    new_content = content.replace(old_text, new_text, 1)
    with open(path, "w") as f:
        f.write(new_content)

    return f"[edit applied] {path} updated."


def run_tests(path: str = ".") -> str:
    """Run pytest under `path` and return pass/fail summary plus output."""
    result = subprocess.run(
        ["pytest", path, "-v"],
        capture_output=True,
        text=True,
    )
    status = "PASSED" if result.returncode == 0 else "FAILED"
    output = result.stdout + result.stderr
    return f"[tests {status}]\n{output}"


def _extract_code_block(text: str) -> str:
    """Pull a ```python ... ``` (or bare ``` ... ```) fenced block out of a
    model response. Handles a closed fence, an unclosed one (generation cut
    off by max_tokens before the closing ```), or no fence at all."""
    closed = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if closed:
        return closed.group(1)
    opened = re.search(r"```(?:python)?\n(.*)", text, re.DOTALL)
    if opened:
        return opened.group(1)
    return text


def _extract_imports(filepath: str) -> str:
    """Return the source file's own top-level import statements, so a
    function whose signature/body references a name like Optional or a
    custom type from that file (e.g. Optional[str], JobListing) actually
    has it available when tested standalone."""
    with open(filepath, "r") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    import_nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    try:
        return ast.unparse(ast.Module(body=import_nodes, type_ignores=[]))
    except Exception:
        return ""


def _strip_and_inject_real_function(
    candidate_code: str,
    function_name: str,
    real_source: str,
    imports_source: str = "",
    enclosing_class: str | None = None,
) -> str | None:
    """Fine-tuned test-gen models tend to reimplement the target function
    (or its class) inline in their test file rather than import it — which
    would mean the test never actually exercises the real code. Parse the
    candidate, delete any top-level def/class that would shadow the real
    one, and prepend the real source instead. Returns None if the candidate
    isn't valid Python at all."""
    try:
        tree = ast.parse(candidate_code)
    except SyntaxError:
        return None

    def shadows_real_code(node) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return True
        if enclosing_class and isinstance(node, ast.ClassDef) and node.name == enclosing_class:
            return True
        return False

    tree.body = [node for node in tree.body if not shadows_real_code(node)]

    try:
        remaining_code = ast.unparse(tree)
    except Exception:
        return None

    prefix = f"{imports_source}\n" if imports_source else ""
    return f"{prefix}{real_source}\n\n{remaining_code}"


def _build_test_prompt(function_name: str, filepath: str, function_source: str, enclosing_class: str | None) -> tuple[str, str]:
    """Build the initial prompt for generate_tests, inlining context the
    fine-tuned model otherwise can't see: (a) any custom type defined in the
    same file whose name is referenced in the function's signature/body,
    and (b) if the function is a class method, the class's own definition
    plus an instruction to instantiate it rather than call the method bare.

    Returns (prompt, real_source_to_inject) — real_source_to_inject is the
    function's own source for a free function, or the enclosing class's
    full source for a method (since a bare method snippet keeps its
    original indentation and isn't valid top-level code on its own).
    """
    context_blocks = []
    referenced_class_names = set()

    for class_name, start, end in find_classes_in_file(filepath):
        if class_name and re.search(rf"\b{re.escape(class_name)}\b", function_source):
            referenced_class_names.add(class_name)

    if enclosing_class:
        referenced_class_names.add(enclosing_class)

    with open(filepath, "r") as f:
        all_lines = f.readlines()

    type_def_sources = []
    enclosing_class_source = None
    for class_name in sorted(referenced_class_names):
        loc = find_class_location(class_name, filepath)
        if loc is None:
            continue
        start, end = loc
        class_source = "".join(all_lines[start - 1 : end])
        context_blocks.append(f"# type referenced by the function below:\n{class_source}")
        type_def_sources.append(class_source)
        if class_name == enclosing_class:
            enclosing_class_source = class_source

    # Any referenced type shown in the prompt must also actually be present
    # in the final injected source, not just visible to the model while it
    # writes its own test — otherwise a type only the model "saw" causes a
    # NameError once we strip the model's own code down to the real
    # function. A method's enclosing class already contains the method
    # itself, so it stands alone; a free function still needs its own
    # source appended alongside any referenced types.
    if enclosing_class:
        real_source_to_inject = enclosing_class_source or function_source
    elif type_def_sources:
        real_source_to_inject = "\n\n".join(type_def_sources + [function_source])
    else:
        real_source_to_inject = function_source

    instructions = "Write pytest-style tests using plain assert statements."
    if enclosing_class:
        instructions += (
            f" `{function_name}` is a method of the `{enclosing_class}` class shown above — "
            f"instantiate `{enclosing_class}(...)` and call `.{function_name}(...)` on that "
            "instance. Do not call it as a bare function."
        )

    context_text = "\n\n".join(context_blocks)
    prompt = (
        f"{context_text}\n\n" if context_text else ""
    ) + f"Create unit tests for the following function:\n\n{function_source}\n\n{instructions}"

    return prompt, real_source_to_inject


def _run_candidate_test(source: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_aegis_gen_test.py", dir=".", delete=False
    ) as f:
        f.write(source)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["pytest", temp_path, "-v"], capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        # A candidate that reads stdin (e.g. testing an interactive
        # function without mocking input) would otherwise hang forever.
        return False, "[timed out after 30s — candidate likely blocked on stdin or an infinite loop]"
    finally:
        os.remove(temp_path)


def generate_tests(function_name: str, n: int = 3) -> str:
    """Generate unit tests for `function_name` using the local fine-tuned
    model (Phase 6): find the function via the symbol index, build a prompt
    with any custom types it needs inlined, then run a sequential fix loop
    (same idea as Phase 4's main-agent fix loop) — generate a candidate,
    AST-check it, run it against the REAL function via pytest, and if it
    fails, feed the exact failure back to the model for the next attempt.
    Stops at the first pass or after `n` attempts."""
    from src.models import generate_local_chat

    location = find_symbol_location(function_name)
    if location is None:
        return f"[generate_tests] no definition found for '{function_name}'."
    filepath, start_line, end_line, enclosing_class = location

    with open(filepath, "r") as f:
        all_lines = f.readlines()
    function_source = "".join(all_lines[start_line - 1 : end_line])
    imports_source = _extract_imports(filepath)

    prompt, real_source_to_inject = _build_test_prompt(function_name, filepath, function_source, enclosing_class)
    messages = [{"role": "user", "content": prompt}]

    attempts = []
    for i in range(1, n + 1):
        response_text = generate_local_chat(messages)
        messages.append({"role": "assistant", "content": response_text})
        code = _extract_code_block(response_text)

        # AST pre-check (fix 2): don't waste a pytest subprocess on code
        # that can't even parse — capture the exact error for feedback.
        try:
            ast.parse(code)
        except SyntaxError as exc:
            failure = f"[SyntaxError] {exc}"
            attempts.append((i, False, failure))
            if i < n:
                messages.append(
                    {
                        "role": "user",
                        "content": f"That code had a syntax error: {exc}. Provide the complete corrected test file.",
                    }
                )
            continue

        cleaned = _strip_and_inject_real_function(
            code, function_name, real_source_to_inject, imports_source, enclosing_class
        )
        if cleaned is None:
            attempts.append((i, False, "[could not process candidate after parsing]"))
            continue

        passed, output = _run_candidate_test(cleaned)
        attempts.append((i, passed, output))
        if passed:
            return f"[generate_tests] candidate {i}/{n} passed for '{function_name}':\n\n{cleaned}"

        if i < n:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That test failed when run against the real function. pytest output:\n"
                        f"{_truncate(output)}\n\nProvide the complete corrected test file."
                    ),
                }
            )

    if not attempts:
        return f"[generate_tests] none of {n} candidates passed for '{function_name}'.\n(no candidates generated)"

    summary = "\n".join(f"candidate {i}: {'PASSED' if ok else 'FAILED'}" for i, ok, _ in attempts)
    all_failures = "\n\n".join(
        f"--- candidate {i} failure ---\n{_truncate(output)}" for i, ok, output in attempts if not ok
    )
    return (
        f"[generate_tests] none of {n} candidates passed for '{function_name}'.\n{summary}\n\n"
        f"{all_failures}"
    )


def run_shell(command: str, cwd: str = ".") -> str:
    """Run a shell command, streaming stdout/stderr live, and return the
    combined output once the process exits."""
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    process.wait()
    output = "".join(output_lines)
    if process.returncode != 0:
        output += f"\n[exit code: {process.returncode}]"
    return output


# --- Tool schemas (Groq / OpenAI-compatible function-calling format) ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a directory, recursively, optionally filtered by a glob pattern (e.g. '*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search. Defaults to current directory."},
                    "pattern": {"type": "string", "description": "Glob pattern to filter filenames. Defaults to '*' (all files)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_symbol",
            "description": "Find where a function or class is defined (file and line range), using the codebase symbol index. Prefer this over reading every file when you know the symbol's name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The exact function or class name to look up."},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a regex pattern, returning matching 'file:line: text' entries. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "directory": {"type": "string", "description": "Directory to search. Defaults to current directory."},
                    "file_pattern": {"type": "string", "description": "Glob pattern to filter which filenames are searched. Defaults to '*.py'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Apply a small patch to a file by replacing an exact block of old text with new "
                "text (search-and-replace). old_text must appear exactly once in the file — "
                "include enough surrounding lines to make it unique. This mutates the file, so "
                "it requires user approval. Prefer this over rewriting whole files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit."},
                    "old_text": {"type": "string", "description": "Exact existing text to replace. Must match exactly once."},
                    "new_text": {"type": "string", "description": "Text to replace old_text with."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest under a path and return whether it passed/failed plus the output. Read-only (doesn't modify files), runs without approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to run pytest against. Defaults to current directory."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_tests",
            "description": (
                "Generate unit tests for a specific function using a local fine-tuned model. "
                "Finds the function via the symbol index, generates several candidate tests, "
                "runs each against the real function via pytest, and returns the first one "
                "that passes (or a plain report if none did). Read-only — doesn't modify files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_name": {"type": "string", "description": "The exact name of the function to generate tests for."},
                    "n": {"type": "integer", "description": "How many candidate tests to try. Defaults to 3."},
                },
                "required": ["function_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return its combined stdout/stderr output. This can mutate the filesystem or system state, so it requires user approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                    "cwd": {"type": "string", "description": "Working directory to run the command in. Defaults to current directory."},
                },
                "required": ["command"],
            },
        },
    },
]

# Tool names that mutate the filesystem or run arbitrary system commands.
# Anything not in this set is treated as read-only and runs without a prompt.
MUTATING_TOOLS = {"run_shell", "edit_file"}

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
    "lookup_symbol": lookup_symbol,
    "grep": grep,
    "edit_file": edit_file,
    "run_tests": run_tests,
    "generate_tests": generate_tests,
    "run_shell": run_shell,
}
