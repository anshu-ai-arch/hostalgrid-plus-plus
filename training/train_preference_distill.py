from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_chosen_dataset(pref_path: str, chosen_path: str):
    rows = []
    with open(pref_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "prompt": item["prompt"],
                "completion": item["chosen"],
                "mode": item["mode"],
                "seed": item["seed"],
                "best_reward": item["best_reward"],
                "worst_reward": item["worst_reward"],
            })

    with open(chosen_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return rows


def format_example(example):
    return {
        "text": (
            "<|user|>\n" + example["prompt"] +
            "\n<|assistant|>\n" + example["completion"]
        )
    }


def main():
    pref_path = os.path.join(REPO_ROOT, "data", "llm_strategy_preferences.jsonl")
    chosen_path = os.path.join(REPO_ROOT, "data", "llm_strategy_chosen_sft.jsonl")
    out_dir = "/content/drive/MyDrive/hostelgrid-work/outputs/energymind-prefdistill-qwen05b"

    rows = build_chosen_dataset(pref_path, chosen_path)
    print("Saved chosen-only dataset:", chosen_path)
    print("Rows:", len(rows))
    if rows:
        print(rows[0])

    dataset = load_dataset("json", data_files=chosen_path, split="train")
    dataset = dataset.map(format_example)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    config = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=30,
        logging_steps=2,
        save_steps=30,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=config,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    trainer.train()
    trainer.model.save_pretrained(out_dir)
    trainer.processing_class.save_pretrained(out_dir)

    print("Saved preference-distilled adapter to:", out_dir)


if __name__ == "__main__":
    main()
