#!/usr/bin/env python3
"""Generate prompts from GSM8K test split (first 25 questions)."""

from datasets import load_dataset

def main():
    # Load GSM8K dataset (MIT-licensed)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    
    # Extract first 25 questions
    prompts = [ex["question"] for ex in ds.select(range(25))]
    
    # Write prompts separated by delimiter
    with open("prompts.txt", "w") as f:
        f.write("\n\n---\n\n".join(prompts))
    
    print(f"Wrote {len(prompts)} prompts to prompts.txt")

if __name__ == "__main__":
    main()

