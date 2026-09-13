#!/usr/bin/env bash
# QLoRA fine-tune: base model is already 4-bit quantized (finetune/models/
# qwen2.5-coder-3b-4bit), so training against it is QLoRA per mlx-lm's own
# definition ("if --model points to a quantized model, training will use
# QLoRA").
set -euo pipefail
cd "$(dirname "$0")/.."

mlx_lm.lora \
    --model finetune/models/qwen2.5-coder-3b-4bit \
    --train \
    --data finetune/data \
    --iters 300 \
    --batch-size 1 \
    --num-layers 4 \
    --grad-checkpoint \
    --adapter-path finetune/adapters
