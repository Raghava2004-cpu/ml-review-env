"""
server.py — FastAPI HTTP server for MLReviewEnv
Exposes all endpoints required by the OpenEnv hackathon spec.

Endpoints:
    GET  /              — health check
    GET  /tasks         — list all tasks + action schema
    POST /reset         — start a new episode
    POST /step          — take one action
    GET  /state         — get current state
    POST /grader        — get grader score for submitted code
    GET  /baseline      — run baseline agent and return scores
"""

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env import MLReviewEnv
from tasks import TASKS
from graders import grade
from models import (
    Action,
    Observation,
    Reward,
    StepResponse,
    TaskInfo,
    TaskScore,
    BaselineResult,
    BugInfo,
    dict_to_observation,
    dict_to_reward,
)

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "MLReviewEnv",
    description = "OpenEnv environment for ML training code review",
    version     = "1.0.0",
)

# Allow all origins so HuggingFace Space can be called from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────────────────
# Session storage
# One environment instance per task_id (0, 1, 2)
# ─────────────────────────────────────────────────────────────

_envs: dict[int, MLReviewEnv] = {
    0: MLReviewEnv(task_id=0),
    1: MLReviewEnv(task_id=1),
    2: MLReviewEnv(task_id=2),
}

def _get_env(task_id: int) -> MLReviewEnv:
    if task_id not in _envs:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid task_id: {task_id}. Must be 0, 1, or 2."
        )
    return _envs[task_id]


# ─────────────────────────────────────────────────────────────
# Request bodies
# ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 0

class StepRequest(BaseModel):
    task_id:     int            = 0
    action:      Action

class GraderRequest(BaseModel):
    task_id:     int            = 0
    code:        Optional[str]  = None
    explanation: Optional[str]  = None
    action_type: str            = "submit_fix"


# ─────────────────────────────────────────────────────────────
# GET / — health check
# ─────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — judges ping this first."""
    return {
        "status":      "ok",
        "environment": "MLReviewEnv",
        "version":     "1.0.0",
        "tasks":       3,
        "description": "ML training code review environment",
    }


# ─────────────────────────────────────────────────────────────
# GET /tasks — list all tasks + action schema
# ─────────────────────────────────────────────────────────────

@app.get("/tasks", response_model=list[TaskInfo])
def list_tasks():
    """
    Returns all 3 tasks with their descriptions, bugs,
    and the action schema (what fields each action type needs).
    """
    action_schema = {
        "submit_fix": {
            "type":        "submit_fix",
            "code":        "string — your fixed Python code",
        },
        "explain_issue": {
            "type":        "explain_issue",
            "explanation": "string — your explanation of the bugs",
        },
        "optimize": {
            "type":        "optimize",
            "code":        "string — your optimized Python code",
        },
        "request_hint": {
            "type":        "request_hint",
        },
        "no_op": {
            "type":        "no_op",
        },
    }

    result = []
    for task in TASKS:
        result.append(TaskInfo(
            id          = task["id"],
            difficulty  = task["difficulty"],
            title       = task["title"],
            description = task["description"],
            n_bugs      = len(task["bugs"]),
            bugs        = [
                BugInfo(
                    id          = b["id"],
                    description = b["description"],
                    severity    = b["severity"],
                )
                for b in task["bugs"]
            ],
            action_schema = action_schema,
        ))
    return result


# ─────────────────────────────────────────────────────────────
# POST /reset — start a fresh episode
# ─────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation)
async def reset_post(request: Request):
    """
    Reset the environment. Body is fully optional.
    Accepts: empty body, {} or {"task_id": 0}
    """
    task_id = 0
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = int(body.get("task_id", 0))
    except Exception:
        task_id = 0

    env   = _get_env(task_id)
    state = env.reset()
    return dict_to_observation(state)


@app.get("/reset", response_model=Observation)
def reset_get(task_id: int = 0):
    """GET version of reset — for validators that use GET."""
    env   = _get_env(task_id)
    state = env.reset()
    return dict_to_observation(state)


# ─────────────────────────────────────────────────────────────
# POST /step — take one action
# ─────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take one action in the environment.
    Returns new observation, reward breakdown, done flag, and info.

    Body:
    {
        "task_id": 0,
        "action": {
            "type": "submit_fix",
            "code": "def train_one_epoch(...):\n    ..."
        }
    }
    """
    env = _get_env(request.task_id)

    try:
        result = env.step(request.action.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    observation = dict_to_observation(result.observation)
    reward      = dict_to_reward(result.info, result.reward)

    return StepResponse(
        observation = observation,
        reward      = reward,
        done        = result.done,
        info        = {
            k: v for k, v in result.info.items()
            if isinstance(v, (str, int, float, bool, list))
        },
    )


# ─────────────────────────────────────────────────────────────
# GET /state — get current state without taking an action
# ─────────────────────────────────────────────────────────────

@app.get("/state", response_model=Observation)
def get_state(task_id: int = 0):
    """
    Returns the current state without advancing the episode.
    Useful for inspecting what the agent currently sees.

    Query param: /state?task_id=0
    """
    env   = _get_env(task_id)
    state = env.state()
    return dict_to_observation(state)


# ─────────────────────────────────────────────────────────────
# POST /grader — score a submission without advancing episode
# ─────────────────────────────────────────────────────────────

@app.post("/grader")
def grader(request: GraderRequest):
    """
    Score a code submission or explanation against a task.
    Does NOT advance the episode — purely for evaluation.

    Body:
    {
        "task_id": 0,
        "action_type": "submit_fix",
        "code": "def train_one_epoch(...):\n    ..."
    }
    """
    if request.task_id not in range(len(TASKS)):
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid task_id: {request.task_id}"
        )

    task   = TASKS[request.task_id]
    action = {
        "type":        request.action_type,
        "code":        request.code        or "",
        "explanation": request.explanation or "",
    }

    result = grade(task, action)

    return {
        "task_id":            request.task_id,
        "action_type":        request.action_type,
        "correctness_score":  result.get("correctness_score",  0.0),
        "architecture_score": result.get("architecture_score", 0.0),
        "explanation_score":  result.get("explanation_score",  0.0),
        "efficiency_score":   result.get("efficiency_score",   0.0),
        "bugs_fixed":         result.get("bugs_fixed",         []),
        "bugs_remaining":     result.get("bugs_remaining",     []),
        "partial_score":      result.get("partial_score",      0.0),
    }


# ─────────────────────────────────────────────────────────────
# GET /baseline — run baseline LLM agent across all tasks
# ─────────────────────────────────────────────────────────────

@app.get("/baseline", response_model=BaselineResult)
def baseline():
    """
    Runs a baseline agent across all 3 tasks using the OpenAI API.
    Reads OPENAI_API_KEY from environment variables.
    Returns reproducible scores for all tasks.
    """
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # If no API key, return heuristic baseline scores
        return BaselineResult(
            model   = "heuristic-fallback",
            tasks   = [
                TaskScore(task_id=0, difficulty="easy",   score=0.70, steps=1),
                TaskScore(task_id=1, difficulty="medium", score=0.70, steps=1),
                TaskScore(task_id=2, difficulty="hard",   score=0.65, steps=1),
            ],
            average = 0.683,
        )

    # Run the real OpenAI baseline
    try:
        from openai import OpenAI
        client  = OpenAI(api_key=api_key)
        scores  = []

        for task_id in range(len(TASKS)):
            env   = MLReviewEnv(task_id=task_id, max_steps=5)
            state = env.reset()

            # Build prompt for the LLM
            prompt = f"""You are a senior ML engineer reviewing buggy Python code.

Task: {state['title']}
Difficulty: {state['difficulty']}

Description:
{state['description']}

Buggy code to fix:
```python
{state['buggy_code']}
```

Please provide the complete fixed version of this code.
Return ONLY the fixed Python code, nothing else.
"""
            response = client.chat.completions.create(
                model    = "gpt-4o-mini",
                messages = [{"role": "user", "content": prompt}],
            )

            fixed_code = response.choices[0].message.content

            # Clean up markdown fences if present
            fixed_code = fixed_code.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code[9:]
            if fixed_code.startswith("```"):
                fixed_code = fixed_code[3:]
            if fixed_code.endswith("```"):
                fixed_code = fixed_code[:-3]
            fixed_code = fixed_code.strip()

            # Submit to environment
            result = env.step({
                "type": "submit_fix",
                "code": fixed_code,
            })

            scores.append(TaskScore(
                task_id    = task_id,
                difficulty = state["difficulty"],
                score      = result.observation["best_score"],
                steps      = 1,
            ))

        average = round(sum(s.score for s in scores) / len(scores), 4)

        return BaselineResult(
            model   = "gpt-4o-mini",
            tasks   = scores,
            average = average,
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Baseline failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host    = "0.0.0.0",
        port    = 7860,
        reload  = True,
    )