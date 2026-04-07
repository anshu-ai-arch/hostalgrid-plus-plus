# app.py — FastAPI wrapper for HF Spaces

import json
import threading
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from env.openenv_api import HostelGridOpenEnv, Action

app = FastAPI(
    title="HostelGrid++",
    description="Human-Aware Energy Optimization Environment",
    version="1.0.0"
)

# ── Health check (judges ping this) ──────────────────────────
@app.get("/")
def root():
    return HTMLResponse("""
    <html>
    <body style="font-family:monospace;padding:40px;background:#0d1117;color:#58a6ff">
        <h1>🏨 HostelGrid++</h1>
        <p>Human-Aware Energy Optimization Environment</p>
        <h3>Endpoints:</h3>
        <ul>
            <li><a href="/docs" style="color:#58a6ff">/docs</a> — API documentation</li>
            <li><a href="/reset" style="color:#58a6ff">/reset</a> — Reset environment</li>
            <li><a href="/state" style="color:#58a6ff">/state</a> — Current state</li>
            <li><a href="/tasks" style="color:#58a6ff">/tasks</a> — List all tasks</li>
            <li><a href="/scores" style="color:#58a6ff">/scores</a> — Latest scores</li>
        </ul>
        <h3>Tasks:</h3>
        <ul>
            <li>🟢 task_easy — Commitment-Aware Energy Allocation</li>
            <li>🟡 task_medium — Fair Enforcement Under Misuse</li>
            <li>🔴 task_hard — Crisis Governance Under Extreme Conditions</li>
        </ul>
    </body>
    </html>
    """)

# ── Environment endpoints ─────────────────────────────────────
environments = {}

@app.get("/reset")
@app.post("/reset")
def reset(task_id: str = "task_easy"):
    env = HostelGridOpenEnv(task_id=task_id)
    environments[task_id] = env
    obs = env.reset()
    return JSONResponse({
        "task_id":     task_id,
        "observation": obs.model_dump(),
        "state":       env.state()
    })

@app.post("/step")
def step(task_id: str = "task_easy", action_id: int = 5):
    if task_id not in environments:
        env = HostelGridOpenEnv(task_id=task_id)
        environments[task_id] = env
        env.reset()
    env = environments[task_id]
    action = Action(action_id=action_id)
    obs, reward, done, info = env.step(action)
    return JSONResponse({
        "task_id":     task_id,
        "observation": obs.model_dump(),
        "reward":      reward.value,
        "done":        done,
        "info":        info,
        "score":       env.score() if done else None
    })

@app.get("/state")
def state(task_id: str = "task_easy"):
    if task_id not in environments:
        env = HostelGridOpenEnv(task_id=task_id)
        environments[task_id] = env
        env.reset()
    return JSONResponse(environments[task_id].state())

@app.get("/tasks")
def tasks():
    return JSONResponse({
        "tasks": [
            {
                "id":         "task_easy",
                "name":       "Commitment-Aware Energy Allocation",
                "difficulty": "easy",
                "max_score":  1.0
            },
            {
                "id":         "task_medium",
                "name":       "Fair Enforcement Under Misuse",
                "difficulty": "medium",
                "max_score":  1.0
            },
            {
                "id":         "task_hard",
                "name":       "Crisis Governance Under Extreme Conditions",
                "difficulty": "hard",
                "max_score":  1.0
            }
        ]
    })

@app.get("/scores")
def scores():
    result = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env = HostelGridOpenEnv(task_id=task_id)
        env.reset()
        done = False
        import random
        while not done:
            _, _, done, _ = env.step(Action(action_id=random.randint(0,5)))
        result[task_id] = env.score()
    avg = sum(result.values()) / len(result)
    result["average"] = round(avg, 4)
    return JSONResponse(result)

def main():
    import app  # or whatever your app's startup logic is
    # If your app already has a launch/run call at the bottom,
    # just wrap it in main() like this:
    pass  # replace with your actual startup code

if __name__ == "__main__":
    main()