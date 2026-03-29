"""
models.py — Pydantic typed models for MLReviewEnv
Required by the OpenEnv spec — all inputs and outputs must be typed.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# ACTION — what the agent sends TO the environment
# ─────────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    What the agent does each step.
    Always requires 'type'. Other fields depend on type.

    Examples:
        {"type": "submit_fix",    "code": "def train..."}
        {"type": "explain_issue", "explanation": "The bug is..."}
        {"type": "optimize",      "code": "def train..."}
        {"type": "request_hint"}
        {"type": "no_op"}
    """
    type: str = Field(
        ...,
        description="Action type: submit_fix | explain_issue | optimize | request_hint | no_op"
    )
    code: Optional[str] = Field(
        None,
        description="Fixed or optimized code. Required for submit_fix and optimize."
    )
    explanation: Optional[str] = Field(
        None,
        description="Explanation of what the bugs are. Required for explain_issue."
    )


# ─────────────────────────────────────────────────────────────
# OBSERVATION — what the agent sees FROM the environment
# ─────────────────────────────────────────────────────────────

class SubScores(BaseModel):
    """Breakdown of best scores achieved so far."""
    correctness:  float = Field(0.0, description="Bug fix correctness (0.0-1.0)")
    architecture: float = Field(0.0, description="Design pattern correctness (0.0-1.0)")
    explanation:  float = Field(0.0, description="Bug explanation quality (0.0-1.0)")
    efficiency:   float = Field(0.0, description="Code efficiency (0.0-1.0)")


class Observation(BaseModel):
    """
    The full state the agent sees after every step.
    This is returned by reset() and inside every StepResult.
    """
    # Task info
    task_id:         int   = Field(..., description="0=easy, 1=medium, 2=hard")
    difficulty:      str   = Field(..., description="easy | medium | hard")
    title:           str   = Field(..., description="Short task title")
    description:     str   = Field(..., description="Full task description with bug hints")
    buggy_code:      str   = Field(..., description="The broken ML code to fix")
    n_bugs:          int   = Field(..., description="Number of bugs in this task")

    # Agent progress
    steps_taken:     int   = Field(..., description="Steps used so far")
    steps_remaining: int   = Field(..., description="Steps left before episode ends")
    hints_used:      int   = Field(..., description="Number of hints requested")
    best_score:      float = Field(..., description="Best total reward achieved this episode")
    best_sub_scores: SubScores = Field(..., description="Best score per dimension")
    done:            bool  = Field(..., description="Is the episode finished?")


# ─────────────────────────────────────────────────────────────
# REWARD — score breakdown returned after each step
# ─────────────────────────────────────────────────────────────

class Reward(BaseModel):
    """
    Detailed reward breakdown for one step.
    Total reward = weighted sum of all sub-scores.
    """
    total:            float = Field(..., description="Total reward for this step (-0.2 to 1.0)")
    correctness:      float = Field(0.0, description="Were the bugs fixed? (weight: 0.45)")
    architecture:     float = Field(0.0, description="Is design pattern correct? (weight: 0.20)")
    explanation:      float = Field(0.0, description="Is explanation correct? (weight: 0.20)")
    efficiency:       float = Field(0.0, description="Is code efficient? (weight: 0.10)")
    speed_bonus:      float = Field(0.0, description="Bonus for solving quickly (weight: 0.05)")
    hint_penalty:     float = Field(0.0, description="Penalty for hints used")
    bugs_fixed:       list  = Field(default_factory=list, description="Bug IDs fixed this step")
    bugs_remaining:   list  = Field(default_factory=list, description="Bug IDs still not fixed")


# ─────────────────────────────────────────────────────────────
# STEP RESULT — full response from step()
# ─────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    """Full response returned by the /step endpoint."""
    observation: Observation
    reward:      Reward
    done:        bool  = Field(..., description="Is episode finished?")
    info:        dict  = Field(default_factory=dict, description="Extra debug info")


# ─────────────────────────────────────────────────────────────
# TASK INFO — returned by /tasks endpoint
# ─────────────────────────────────────────────────────────────

class BugInfo(BaseModel):
    """Description of one bug in a task."""
    id:          str = Field(..., description="Unique bug identifier")
    description: str = Field(..., description="What the bug is")
    severity:    str = Field(..., description="critical | high | medium | low")


class TaskInfo(BaseModel):
    """Summary of one task returned by /tasks endpoint."""
    id:          int       = Field(..., description="Task ID (0, 1, or 2)")
    difficulty:  str       = Field(..., description="easy | medium | hard")
    title:       str       = Field(..., description="Short task title")
    description: str       = Field(..., description="Full task description")
    n_bugs:      int       = Field(..., description="Number of bugs to fix")
    bugs:        list[BugInfo] = Field(..., description="List of bugs in this task")
    action_schema: dict    = Field(..., description="Required fields for each action type")


# ─────────────────────────────────────────────────────────────
# BASELINE RESULT — returned by /baseline endpoint
# ─────────────────────────────────────────────────────────────

class TaskScore(BaseModel):
    """Score for one task in the baseline run."""
    task_id:    int   = Field(..., description="Task ID")
    difficulty: str   = Field(..., description="easy | medium | hard")
    score:      float = Field(..., description="Final score (0.0-1.0)")
    steps:      int   = Field(..., description="Steps taken")


class BaselineResult(BaseModel):
    """Full baseline result returned by /baseline endpoint."""
    model:        str             = Field(..., description="Model used for baseline")
    tasks:        list[TaskScore] = Field(..., description="Score per task")
    average:      float           = Field(..., description="Average score across all tasks")


# ─────────────────────────────────────────────────────────────
# Helper: convert env state dict → Observation model
# ─────────────────────────────────────────────────────────────

def dict_to_observation(state: dict) -> Observation:
    """Convert the raw dict from env.state() into a typed Observation."""
    sub = state.get("best_sub_scores", {})
    return Observation(
        task_id         = state["task_id"],
        difficulty      = state["difficulty"],
        title           = state["title"],
        description     = state["description"],
        buggy_code      = state["buggy_code"],
        n_bugs          = state["n_bugs"],
        steps_taken     = state["steps_taken"],
        steps_remaining = state["steps_remaining"],
        hints_used      = state["hints_used"],
        best_score      = state["best_score"],
        best_sub_scores = SubScores(
            correctness  = sub.get("correctness",  0.0),
            architecture = sub.get("architecture", 0.0),
            explanation  = sub.get("explanation",  0.0),
            efficiency   = sub.get("efficiency",   0.0),
        ),
        done = state["done"],
    )


def dict_to_reward(info: dict, reward: float) -> Reward:
    """Convert a grade result dict into a typed Reward model."""
    return Reward(
        total           = reward,
        correctness     = info.get("correctness_score",  0.0),
        architecture    = info.get("architecture_score", 0.0),
        explanation     = info.get("explanation_score",  0.0),
        efficiency      = info.get("efficiency_score",   0.0),
        speed_bonus     = 0.0,
        hint_penalty    = 0.0,
        bugs_fixed      = info.get("bugs_fixed",      []),
        bugs_remaining  = info.get("bugs_remaining",  []),
    )