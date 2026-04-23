from __future__ import annotations

import os
import random
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_example(tokenizer, device, prompt, completion):
    prompt_text = "<|user|>\n" + prompt + "\n<|assistant|>\n"
    full_text = prompt_text + completion

    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)
    labels = input_ids.clone()

    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    return input_ids, attention_mask, labels


def main():
    rollout_path = os.path.join(REPO_ROOT, "data", "llm_strategy_grpo_rollouts.jsonl")
    out_dir = "/content/drive/MyDrive/hostelgrid-work/outputs/energymind-grpo-lite-qwen05b"

    train_rows = load_dataset("json", data_files=rollout_path, split="train")
    train_rows = [row for row in train_rows if len(row["completion"].strip()) > 0]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    epochs = 1
    grad_accum = 4
    step = 0

    for epoch in range(epochs):
        random.shuffle(train_rows)

        for idx, row in enumerate(train_rows):
            input_ids, attention_mask, labels = build_example(
                tokenizer,
                device,
                row["prompt"],
                row["completion"],
            )
            advantage = torch.tensor(row["advantage"], dtype=torch.float32, device=device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = advantage * outputs.loss
            loss.backward()

            if (idx + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"step={step} loss={loss.item():.4f} advantage={advantage.item():.4f}")

    optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print("Saved GRPO-lite adapter to:", out_dir)


if __name__ == "__main__":
    main()
