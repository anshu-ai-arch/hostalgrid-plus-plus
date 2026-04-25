from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    base_path = os.path.join(REPO_ROOT, "data", "llm_strategy_preferences.jsonl")
    grouped_path = os.path.join(REPO_ROOT, "data", "llm_strategy_grouped_preferences.jsonl")
    out_path = os.path.join(REPO_ROOT, "data", "llm_strategy_merged_preferences_hard.jsonl")

    rows = []
    seen = set()

    for path in [base_path, grouped_path]:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("mode") != "hard":
                    continue
                key = (row.get("prompt"), row.get("chosen"), row.get("rejected"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print("Saved:", out_path)
    print("Rows:", len(rows))
    if rows:
        print(rows[0])


if __name__ == "__main__":
    main()
