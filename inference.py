import os
import time
from openai import OpenAI

from env.openenv_api import HostelGridOpenEnv, Action
from training.llm_strategy import execute_strategy, make_strategy_prompt, parse_strategy

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    base_url=API_BASE_URL,
)

TASKS = ["task_easy", "task_medium", "task_hard"]


def get_strategy_from_llm(env, task_id: str) -> dict:
    prompt = make_strategy_prompt(env._env)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        strategy, valid = parse_strategy(text)
        if not valid:
            strategy = "priority_safe"
        return {
            "strategy": strategy,
            "valid": valid,
            "raw": text,
        }
    except Exception as exc:
        return {
            "strategy": "priority_safe",
            "valid": False,
            "raw": f"fallback:{exc}",
        }


def run_task(task_id: str) -> float:
    env = HostelGridOpenEnv(task_id=task_id)
    obs = env.reset()

    print(f"[START] task={task_id}", flush=True)

    done = False
    step = 0

    while not done:
        step += 1

        llm_result = get_strategy_from_llm(env, task_id)
        room_actions = execute_strategy(env._env, llm_result["strategy"])

        obs, reward, done, info = env.step(Action(room_actions=room_actions))

        print(
            f"[STEP] step={step} "
            f"strategy={llm_result['strategy']} "
            f"valid={llm_result['valid']} "
            f"reward={reward.value:.4f} "
            f"score={env.score():.4f}",
            flush=True,
        )

    score = env.score()

    print(f"[END] task={task_id} score={score:.4f} steps={step}", flush=True)
    return score


def main():
    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running {task_id} with strategy-planning inference...", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_task(task_id)
        scores[task_id] = score
        print(f"\n{task_id} Score: {score:.4f}", flush=True)
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)

    print(f"\n{'='*60}", flush=True)
    print("Final Scores:", flush=True)
    for t, s in scores.items():
        print(f"  {t}: {s:.4f}", flush=True)
    print(f"  average: {avg:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
