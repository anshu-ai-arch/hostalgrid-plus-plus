from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from huggingface_hub import HfApi

BASE_MODEL = os.environ.get("HF_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def build_chosen_dataset(pref_path: str, chosen_path: str):
    rows = []
    with open(pref_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "prompt": item["prompt"],
                "completion": item["chosen"],
            })

    with open(chosen_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return rows


def format_example(example):
    return {
        "text": "<|user|>\n" + example["prompt"] + "\n<|assistant|>\n" + example["completion"]
    }


def main():
    pref_path = os.environ.get("HF_HARD_PREF_PATH", os.path.join(REPO_ROOT, "data", "llm_strategy_preferences_hard_train.jsonl"))
    chosen_path = os.path.join(REPO_ROOT, "data", "llm_strategy_hard_train_chosen_sft.jsonl")
    out_dir = os.environ.get(
        "HF_JOB_OUT_DIR",
        os.path.join(REPO_ROOT, "experiment_runs", "2026-04-25_dqn_campaign", "post_training", "checkpoints", "energymind-prefdistill-hard-qwen05b"),
    )

    rows = build_chosen_dataset(pref_path, chosen_path)
    print("Saved chosen-only dataset:", chosen_path)
    print("Rows:", len(rows))

    dataset = load_dataset("json", data_files=chosen_path, split="train")
    dataset = dataset.map(format_example)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=float(os.environ.get("HF_LR", "8e-5")),
        max_steps=int(os.environ.get("HF_MAX_STEPS", "120")),
        logging_steps=5,
        save_steps=80,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=BASE_MODEL,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    trainer.train()

    log_history = trainer.state.log_history
    history_path = os.path.join(out_dir, "trainer_log_history.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    metric_rows = [row for row in log_history if "loss" in row]
    if metric_rows:
        df = pd.DataFrame(metric_rows)
        csv_path = os.path.join(out_dir, "training_metrics.csv")
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(df["step"], df["loss"], marker="o", linewidth=2.5, color="#d2553f")
        plt.title("Hard Preference Distillation Training Loss", fontsize=14, weight="bold")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "training_loss_curve.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        print("Saved training metrics to:", csv_path)
        print("Saved training curve to:", plot_path)

    trainer.model.save_pretrained(out_dir)
    trainer.processing_class.save_pretrained(out_dir)

    repo_id = os.environ.get("HF_UPLOAD_REPO_ID")
    if repo_id:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="model")
        print("Uploaded adapter to HF repo:", repo_id)

    print("Saved hard preference-distilled adapter to:", out_dir)


if __name__ == "__main__":
    main()
