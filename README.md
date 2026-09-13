# Aegis

A coding agent built from scratch — no agent framework magic, no black boxes. Aegis is a from-first-principles implementation of the core patterns behind tools like Claude Code and Cursor: a ReAct loop, permission-gated tool execution, patch-based editing, AST-based codebase indexing, parallel research subagents, and a locally fine-tuned model for a narrow, verifiable task.

This project exists to *understand* those patterns by building each one, one at a time, on real code — not to import a library that hides them.

## What it does

Point Aegis at any Python project and talk to it in plain English:

```
$ python -m src.main /path/to/your/project
Aegis ready. Working directory: /path/to/your/project
> where is score_and_dedup defined?
> change the signature of get_calls() function to accept a new argument 
> there's a failing test, find and fix it
> how does the scoring logic connect to the notification logic?
```

- **Reads and edits code** — patch-based edits (exact search-and-replace), not full-file rewrites, so a small local model's mistake stays small and reviewable.
- **Asks before it mutates anything** — every file write or shell command pauses for your `y`/`n`, with an auto-deny timeout. Read-only exploration never interrupts you.
- **Knows the codebase structurally** — a tree-sitter + SQLite symbol index answers "where is X defined" instantly, without re-reading every file, and without the cost or staleness of vector embeddings.
- **Fixes its own mistakes** — after an edit, tests run automatically; failures get fed back for another attempt, with a repetition guard so it gives up cleanly instead of looping forever.
- **Parallelizes real research** — a broad question like "how does A connect to B" fans out multiple read-only subagents across the codebase concurrently (LangGraph's `Send` API) and synthesizes their findings, instead of one long serial search.
- **Runs a fine-tuned local model for one narrow job** — generating unit tests — entirely on-device via Apple's MLX, with a deterministic pytest verifier keeping only what actually passes. See below.

## Architecture

```
src/
├── main.py         CLI entrypoint — plain input()/print(), no TUI
├── graph.py        LangGraph StateGraph: the core agent loop + the research fan-out
├── tools.py        read_file, edit_file, run_shell, run_tests, grep, lookup_symbol, generate_tests
├── permissions.py  the approval gate — mutating tools pause, read-only tools don't
├── indexer.py      tree-sitter parsing + SQLite symbol cache
└── models.py       Groq API wrapper (the main loop) + the local MLX model wrapper

finetune/
├── build_dataset.py   curates the fine-tuning dataset
├── train.sh           the QLoRA fine-tune command
└── models/            (gitignored) your converted/quantized/fine-tuned models live here
```

The main loop runs on [Groq](https://console.groq.com/) (`qwen/qwen3.6-27b` by default — fast, free-tier friendly, and one of the few hosted open models with reliable forced tool-calling). The local model is separate and only ever gets one job.

## Getting started

```bash
git clone <this-repo>
cd Aegis
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then add your Groq API key to .env 
```

```bash
python -m src.main /path/to/any/python/project
```

That's the whole main agent — no local model required for any of this.

## The local model: fine-tuning a 3B test-writer with MLX

The main agent handles everything general. But one task — *given a function, write a passing unit test for it* — has real ground truth (a test either passes or it doesn't), which makes it a good candidate for a small, cheap, fully local specialist model instead of another hosted API call.

**What was built, end to end, on an M3 Mac:**

- Curated a filtered sample of the public [`erishabh/unit-test-v1`](https://huggingface.co/datasets/erishabh/unit-test-v1) dataset (real Python functions paired with real pytest tests).
- Fine-tuned `Qwen2.5-Coder-3B-Instruct` with **QLoRA** via [`mlx-lm`](https://github.com/ml-explore/mlx-lm) — training against an already-4-bit-quantized base, entirely on-device, no cloud GPU.
- Quantized to 4-bit and **measured the real memory difference**, not an assumed one: **6.24 GB → 1.85 GB peak memory, a 70% reduction**, verified with `mlx.core.get_peak_memory()` on an actual generation pass.
- Wired it into a **test-time-compute pipeline**: generate a candidate test, parse it, and run it against the *real* function with pytest → if it fails, feed the exact failure back to the model for another attempt (up to 3 tries). keep the first one that genuinely passes. This is a small, deterministic-verifier version of the test-time-scaling idea behind papers like recursive tournament voting and parallel-distill-refine — simpler, because "does the test pass" is real ground truth, not a comparative judgment call.
- Built context-injection on top of the existing tree-sitter indexer: before asking the model to write a test, it automatically finds and inlines any custom type the target function needs, and detects when the function is a class method so the model knows to instantiate the class instead of calling it bare.


- The best-of-3 verifier pipeline **outperformed a single blind attempt** on every function tested — the deterministic-verifier idea works, measurably, not just in theory.
- Context injection **eliminated an entire class of failure**: functions referencing a locally-defined custom type used to fail with `NameError` every time; after inlining the type definition automatically, that failure mode disappeared, including at least one function that went from *guaranteed failure* to a full clean pass.
- The AST pre-check and sequential feedback loop are directly observable working: a syntax error in attempt 1 gets caught before wasting a test run, fed back verbatim, and attempt 2 visibly self-corrects.
- The honest limitation: a 3B model, even fine-tuned, is still a small model — its remaining failures are mostly about raw generation reliability (a missed import in its own scaffolding, a wrong guess at an internal sentinel value) rather than missing context. That's a real, useful finding about where a fine-tuned specialist model like this is and isn't ready to be trusted unsupervised — and knowing that boundary precisely is the actual engineering contribution here.

### Using it

`generate_tests` is a normal Aegis tool — just ask:

```
> generate tests for score_and_dedup
```

**To point it at your own model** (your own fine-tune, or someone else's MLX-converted model), set one environment variable:

```bash
# in .env
AEGIS_LOCAL_MODEL_PATH=/path/to/any/mlx/model/directory
```

**To reproduce the fine-tune yourself:**

```bash
pip install mlx-lm huggingface_hub datasets   

python finetune/build_dataset.py            

mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-3B-Instruct \
  --mlx-path finetune/models/qwen2.5-coder-3b-4bit -q

bash finetune/train.sh                        
mlx_lm.fuse --model finetune/models/qwen2.5-coder-3b-4bit \
  --adapter-path finetune/adapters \
  --save-path finetune/models/qwen2.5-coder-3b-testgen
```

Requires Apple Silicon (MLX is Metal-only). `finetune/train.sh`'s batch size and layer count are tuned for 16GB of unified memory — turn them up if you have more.

