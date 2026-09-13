

import json
import random

from datasets import load_dataset

SEED = 0
SAMPLE_SIZE = 180
VALID_FRACTION = 0.1

MIN_COMPLETION_LINES = 3
MIN_COMPLETION_CHARS = 60
MAX_COMBINED_CHARS = 6000


def is_valid_row(row: dict) -> bool:
    prompt = row["prompt"]
    completion = row["completion"]

    if not prompt or not completion:
        return False
    if "assert" not in completion:
        return False
    if len(completion.strip().splitlines()) < MIN_COMPLETION_LINES:
        return False
    if len(completion.strip()) < MIN_COMPLETION_CHARS:
        return False
    if len(prompt) + len(completion) > MAX_COMBINED_CHARS:
        return False
    return True


def main():
    dataset = load_dataset("erishabh/unit-test-v1", split="train")
    print(f"loaded {len(dataset)} rows")

    filtered = [row for row in dataset if is_valid_row(row)]
    print(f"{len(filtered)} rows passed filtering")

    rng = random.Random(SEED)
    sample = rng.sample(filtered, min(SAMPLE_SIZE, len(filtered)))
    rng.shuffle(sample)

    split_at = int(len(sample) * (1 - VALID_FRACTION))
    train_rows, valid_rows = sample[:split_at], sample[split_at:]

    for name, rows in [("train", train_rows), ("valid", valid_rows)]:
        path = f"finetune/data/{name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps({"prompt": row["prompt"], "completion": row["completion"]}) + "\n")
        print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
