"""Groq API wrapper. Groq's chat-completions API is OpenAI-compatible, not
Anthropic-compatible: tool calls come back as a list under
`message.tool_calls`, each with `.function.name` / `.function.arguments`
(a JSON string) — not Anthropic's `content` list of `type: "tool_use"`
blocks. See https://console.groq.com/docs/tool-use.
"""

import os

from dotenv import load_dotenv
from groq import Groq

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

DEFAULT_MODEL = os.environ.get("AEGIS_MODEL", "qwen/qwen3.6-27b")

_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def call_model(messages: list[dict], tools: list[dict] | None = None, model: str = None):
    """Call the Groq chat-completions API and return the raw response message.

    reasoning_format="parsed" keeps `message.content` as the clean final
    answer (needed for reliable tool-call parsing) while `message.reasoning`
    holds the model's thinking trace separately, per Phase 0's fix.
    """
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "reasoning_format": "parsed",
        "reasoning_effort": "none",
        "max_completion_tokens": 800,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message


# --- Phase 6: local fine-tuned model, one narrow job (test generation) ---
# Lazy-loaded: importing mlx_lm and loading weights is slow and only needed
# when generate_tests is actually called, not on every Aegis startup.

LOCAL_TESTGEN_MODEL_PATH = os.environ.get(
    "AEGIS_LOCAL_MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "finetune", "models", "qwen2.5-coder-3b-testgen"),
)

_local_model = None
_local_tokenizer = None


def _load_local_model():
    global _local_model, _local_tokenizer
    if _local_model is None:
        from mlx_lm import load

        _local_model, _local_tokenizer = load(LOCAL_TESTGEN_MODEL_PATH)
    return _local_model, _local_tokenizer


def generate_local(prompt: str, n: int = 1, temperature: float = 0.7, max_tokens: int = 900) -> list[str]:
    """Generate `n` candidate completions from the local fine-tuned model,
    sampling with `temperature` so repeated calls vary (needed for the
    best-of-N test generation pipeline)."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_local_model()
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)

    return [
        generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
        for _ in range(n)
    ]


def generate_local_chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 900) -> str:
    """Generate one completion from the local fine-tuned model given a full
    conversation (system/user/assistant turns), not just a single prompt.
    Used by the sequential fix-loop in generate_tests, where each retry
    needs to see its own previous attempt and failure — the same idea as
    Phase 4's main-agent fix loop, applied to the local model."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_local_model()
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    sampler = make_sampler(temp=temperature)

    return generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
