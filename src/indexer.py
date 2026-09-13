"""AST-based codebase indexing via tree-sitter, cached in SQLite.

Chosen over vector-embedding RAG: symbol lookups here need exact structural
answers ("where is X defined") with deterministic guarantees, not fuzzy
semantic search. AST parsing gives exact function/class locations for free;
embeddings would need an embedding index and still just approximate this.
Cache is keyed by (filepath, mtime) so unchanged files are never re-parsed.
"""

import os
import sqlite3

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

SYMBOL_NODE_TYPES = ("function_definition", "class_definition")

# Directories never worth indexing: vendored deps and VCS/cache metadata,
# not project code.
EXCLUDED_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".pytest_cache"}

_LANGUAGE = Language(tspython.language())


def _make_parser() -> Parser:
    return Parser(_LANGUAGE)


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            filepath TEXT PRIMARY KEY,
            mtime REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            filepath TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            enclosing_class TEXT
        )
        """
    )
    # Migrate older cache files created before enclosing_class existed.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(symbols)")}
    if "enclosing_class" not in existing_cols:
        conn.execute("ALTER TABLE symbols ADD COLUMN enclosing_class TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
    conn.commit()


def _extract_symbols(source: bytes, parser: Parser) -> list[tuple[str, str, int, int, str | None]]:
    """Return (name, kind, start_line, end_line, enclosing_class) tuples,
    lines 1-indexed. enclosing_class is the name of the class a function is
    nested directly under (a method), or None for a free function/a class
    itself — needed so generate_tests (Phase 6) can tell a method from a
    free function and instantiate its class instead of calling it bare."""
    tree = parser.parse(source)
    symbols = []

    def walk(node, enclosing_class):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = name_node.text.decode() if name_node is not None else None
            if class_name is not None:
                symbols.append((class_name, "class", node.start_point[0] + 1, node.end_point[0] + 1, None))
            for child in node.children:
                walk(child, class_name)
            return

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                symbols.append(
                    (
                        name_node.text.decode(),
                        "function",
                        node.start_point[0] + 1,
                        node.end_point[0] + 1,
                        enclosing_class,
                    )
                )
            # A function nested inside another function is not a method of
            # the enclosing class, even if that function is itself a method.
            for child in node.children:
                walk(child, None)
            return

        for child in node.children:
            walk(child, enclosing_class)

    walk(tree.root_node, None)
    return symbols


def index_repo(root: str, db_path: str) -> None:
    """Walk `root`, parse every .py file, and refresh the SQLite cache at
    `db_path`. Files whose mtime hasn't changed since the last index are
    skipped entirely (neither re-parsed nor re-queried)."""
    conn = sqlite3.connect(db_path)
    _init_db(conn)
    parser = _make_parser()

    cached_mtimes = dict(conn.execute("SELECT filepath, mtime FROM files"))
    seen_files = set()

    for dirpath, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            filepath = os.path.relpath(os.path.join(dirpath, filename), root)
            seen_files.add(filepath)
            mtime = os.path.getmtime(os.path.join(dirpath, filename))

            if cached_mtimes.get(filepath) == mtime:
                continue  # unchanged, skip re-parsing

            with open(os.path.join(dirpath, filename), "rb") as f:
                source = f.read()
            symbols = _extract_symbols(source, parser)

            conn.execute("DELETE FROM symbols WHERE filepath = ?", (filepath,))
            conn.executemany(
                "INSERT INTO symbols (name, kind, filepath, start_line, end_line, enclosing_class) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (name, kind, filepath, start, end, enclosing_class)
                    for name, kind, start, end, enclosing_class in symbols
                ],
            )
            conn.execute(
                "INSERT INTO files (filepath, mtime) VALUES (?, ?) "
                "ON CONFLICT(filepath) DO UPDATE SET mtime = excluded.mtime",
                (filepath, mtime),
            )

    # drop entries for files that no longer exist
    for filepath in cached_mtimes.keys() - seen_files:
        conn.execute("DELETE FROM symbols WHERE filepath = ?", (filepath,))
        conn.execute("DELETE FROM files WHERE filepath = ?", (filepath,))

    conn.commit()
    conn.close()


def find_symbol_location(name: str, db_path: str = ".aegis_index.sqlite3") -> tuple[str, int, int, str | None] | None:
    """Return (filepath, start_line, end_line, enclosing_class) for the
    first definition of `name`, or None if not found. Structured
    counterpart to lookup_symbol, for callers that need to actually read
    the function's source (e.g. generate_tests in Phase 6)."""
    index_repo(".", db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT filepath, start_line, end_line, enclosing_class FROM symbols WHERE name = ? LIMIT 1",
        (name,),
    ).fetchone()
    conn.close()

    return tuple(row) if row else None


def find_class_location(class_name: str, filepath: str, db_path: str = ".aegis_index.sqlite3") -> tuple[int, int] | None:
    """Return (start_line, end_line) for a class named `class_name` defined
    in `filepath`, or None if not found."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT start_line, end_line FROM symbols WHERE name = ? AND kind = 'class' AND filepath = ? LIMIT 1",
        (class_name, filepath),
    ).fetchone()
    conn.close()
    return tuple(row) if row else None


def find_classes_in_file(filepath: str, db_path: str = ".aegis_index.sqlite3") -> list[tuple[str, int, int]]:
    """Return (name, start_line, end_line) for every class defined in
    `filepath` — used to detect custom types referenced in a function's
    signature (Phase 6 generate_tests context injection)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name, start_line, end_line FROM symbols WHERE kind = 'class' AND filepath = ?",
        (filepath,),
    ).fetchall()
    conn.close()
    return [tuple(row) for row in rows]


def lookup_symbol(name: str, db_path: str = ".aegis_index.sqlite3") -> str:
    """Return the file and line range where `name` is defined, using the
    SQLite symbol cache. Re-indexes the current directory first so results
    reflect any edits made since the last lookup."""
    index_repo(".", db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name, kind, filepath, start_line, end_line FROM symbols WHERE name = ?",
        (name,),
    ).fetchall()
    conn.close()

    if not rows:
        return f"[lookup_symbol] no definition found for '{name}'."

    lines = [
        f"{kind} '{sym_name}' defined in {filepath}:{start}-{end}"
        for sym_name, kind, filepath, start, end in rows
    ]
    return "\n".join(lines)
