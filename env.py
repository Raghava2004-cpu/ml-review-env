"""
env.py — MLReviewEnv: Core environment file
Implements the full OpenEnv API: reset() / step() / state()
"""

import copy
from dataclasses import dataclass, field
from typing import Any

from tasks import TASKS
from graders import grade


# ─────────────────────────────────────────────────────────────
# StepResult — what step() returns every time
# ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    observation: dict   # current state the agent sees
    reward: float       # score for this step (-0.2 to 1.0)
    done: bool          # is the episode finished?
    info: dict          # extra details (bugs fixed, hints, etc.)


# ─────────────────────────────────────────────────────────────
# MLReviewEnv — main environment class
# ─────────────────────────────────────────────────────────────

class MLReviewEnv:
    """
    An environment where an AI agent reviews and fixes
    senior-level ML training code bugs.

    Reward breakdown (always between -0.2 and 1.0):
        0.45 x correctness    — are the bugs actually fixed?
        0.20 x architecture   — is the design pattern correct?
        0.20 x explanation    — does the agent explain WHY?
        0.10 x efficiency     — no memory waste, best practices?
        0.05 x speed bonus    — solving faster = tiny extra reward
        0.04 x hints penalty  — each hint costs a little reward

    Partial progress:
        Every individual bug fix contributes to correctness score.
        Agent gets reward signal even if it only fixes 1 out of 3 bugs.
    """

    # Valid action types the agent can use
    VALID_ACTIONS = {
        "submit_fix",     # submit corrected code
        "explain_issue",  # explain what the bugs are and why
        "optimize",       # submit code with performance improvements
        "request_hint",   # get a hint (costs -0.04 reward)
        "no_op",          # do nothing (costs -0.02 reward)
    }

    # How much each dimension contributes to total reward
    REWARD_WEIGHTS = {
        "correctness":  0.45,
        "architecture": 0.20,
        "explanation":  0.20,
        "efficiency":   0.10,
        "speed":        0.05,
    }

    HINT_PENALTY = 0.04
    NOOP_PENALTY = 0.02

    def __init__(self, task_id: int = 0, max_steps: int = 15):
        """
        Create the environment for a specific task.

        Args:
            task_id:   0 = easy, 1 = medium, 2 = hard
            max_steps: maximum steps before episode ends
        """
        if task_id not in range(len(TASKS)):
            raise ValueError(f"task_id must be 0, 1, or 2. Got: {task_id}")

        self.task_id   = task_id
        self.max_steps = max_steps

        # Internal state — reset() will reinitialise all of these
        self._task        = copy.deepcopy(TASKS[task_id])
        self._steps       = 0
        self._hints_used  = 0
        self._done        = False
        self._best_score  = 0.0
        self._best_sub    = {
            "correctness":  0.0,
            "architecture": 0.0,
            "explanation":  0.0,
            "efficiency":   0.0,
        }
        self._history: list[dict] = []

    # ─────────────────────────────────────────────────────────
    # reset() — start a fresh episode
    # ─────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """
        Reset the environment to its initial state.
        Call this at the start of every new episode.
        Returns the starting observation.
        """
        self._task       = copy.deepcopy(TASKS[self.task_id])
        self._steps      = 0
        self._hints_used = 0
        self._done       = False
        self._best_score = 0.0
        self._best_sub   = {k: 0.0 for k in self._best_sub}
        self._history    = []
        return self.state()

    # ─────────────────────────────────────────────────────────
    # step() — take one action
    # ─────────────────────────────────────────────────────────

    def step(self, action: dict) -> StepResult:
        """
        Take one action in the environment.

        action must be a dict with at least:
            { "type": "submit_fix", "code": "..." }
        or
            { "type": "explain_issue", "explanation": "..." }
        or
            { "type": "request_hint" }
        or
            { "type": "no_op" }

        Returns StepResult(observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError(
                "Episode is already finished. Call reset() to start again."
            )

        action_type = action.get("type", "no_op")

        if action_type not in self.VALID_ACTIONS:
            raise ValueError(
                f"Unknown action type: '{action_type}'\n"
                f"Valid types: {sorted(self.VALID_ACTIONS)}"
            )

        self._steps += 1
        info: dict[str, Any] = {
            "step":        self._steps,
            "action_type": action_type,
        }

        # ── Handle each action type ───────────────────────────

        if action_type == "request_hint":
            reward, info = self._give_hint(info)

        elif action_type == "no_op":
            reward = -self.NOOP_PENALTY
            info["message"] = "No-op — a step was wasted."

        else:
            # submit_fix / explain_issue / optimize
            grade_result = grade(self._task, action)
            reward       = self._compute_reward(grade_result)

            # Track best sub-scores independently
            # (agent can improve explanation on one step, code on another)
            for key in self._best_sub:
                score_key = f"{key}_score"
                if score_key in grade_result:
                    self._best_sub[key] = max(
                        self._best_sub[key],
                        grade_result[score_key]
                    )

            self._best_score = max(self._best_score, reward)

            info.update(grade_result)
            info["reward"]         = round(reward, 4)
            info["bugs_fixed"]     = grade_result.get("bugs_fixed", [])
            info["bugs_remaining"] = grade_result.get("bugs_remaining", [])

            # Episode is solved when all bugs fixed + architecture correct
            if (grade_result.get("correctness_score", 0) >= 1.0 and
                    grade_result.get("architecture_score", 0) >= 0.8):
                self._done = True
                info["message"] = "✅ All bugs fixed with correct architecture!"

        # ── Check step limit ──────────────────────────────────

        if self._steps >= self.max_steps and not self._done:
            self._done = True
            info.setdefault(
                "message",
                f"⏱ Out of steps. Best score: {self._best_score:.3f}"
            )

        # ── Save to history ───────────────────────────────────

        self._history.append({
            "step":        self._steps,
            "action_type": action_type,
            "reward":      reward,
            "best_score":  self._best_score,
        })

        return StepResult(
            observation = self.state(),
            reward      = round(reward, 4),
            done        = self._done,
            info        = info,
        )

    # ─────────────────────────────────────────────────────────
    # state() — what the agent currently sees
    # ─────────────────────────────────────────────────────────

    def state(self) -> dict:
        """
        Returns the current observable state of the environment.
        The agent uses this to decide its next action.
        """
        return {
            # Task information
            "task_id":          self.task_id,
            "difficulty":       self._task["difficulty"],
            "title":            self._task["title"],
            "description":      self._task["description"],
            "buggy_code":       self._task["buggy_code"],
            "n_bugs":           len(self._task["bugs"]),

            # Agent progress
            "steps_taken":      self._steps,
            "steps_remaining":  self.max_steps - self._steps,
            "hints_used":       self._hints_used,
            "best_score":       round(self._best_score, 4),
            "best_sub_scores":  {k: round(v, 4) for k, v in self._best_sub.items()},
            "done":             self._done,
        }

    # ─────────────────────────────────────────────────────────
    # Reward computation
    # ─────────────────────────────────────────────────────────

    def _compute_reward(self, grade_result: dict) -> float:
        """
        Weighted reward with partial progress signals.
        Result is always between -0.2 and 1.0.
        """
        w = self.REWARD_WEIGHTS

        r = (
            w["correctness"]  * grade_result.get("correctness_score",  0.0) +
            w["architecture"] * grade_result.get("architecture_score", 0.0) +
            w["explanation"]  * grade_result.get("explanation_score",  0.0) +
            w["efficiency"]   * grade_result.get("efficiency_score",   0.0)
        )

        # Speed bonus: only if fully solved, reward solving early
        if grade_result.get("correctness_score", 0) >= 1.0:
            steps_left   = self.max_steps - self._steps
            speed_frac   = steps_left / self.max_steps
            r += w["speed"] * speed_frac

        # Hint penalty
        r -= self._hints_used * self.HINT_PENALTY

        return round(min(1.0, max(-0.2, r)), 4)

    # ─────────────────────────────────────────────────────────
    # Hint system
    # ─────────────────────────────────────────────────────────

    def _give_hint(self, info: dict) -> tuple[float, dict]:
        hints = self._task.get("hints", [])

        if not hints:
            info["message"] = "No hints available for this task."
            return -self.NOOP_PENALTY, info

        # Give hints one at a time in order
        hint_idx = min(self._hints_used, len(hints) - 1)
        self._hints_used += 1

        info["hint"]       = hints[hint_idx]
        info["hints_used"] = self._hints_used
        info["message"]    = (
            f"💡 Hint {self._hints_used}/{len(hints)} "
            f"(cost: -{self.HINT_PENALTY} reward)"
        )

        return -self.HINT_PENALTY, info

    # ─────────────────────────────────────────────────────────
    # render() — human readable display
    # ─────────────────────────────────────────────────────────

    def render(self) -> str:
        """Print a readable summary of the current state."""
        s    = self.state()
        bar  = "█" * int(s["best_score"] * 20) + "░" * (20 - int(s["best_score"] * 20))

        lines = [
            "┌" + "─" * 54 + "┐",
            f"│  Task {s['task_id']} [{s['difficulty'].upper()}] — {s['title'][:30]:<30}  │",
            "├" + "─" * 54 + "┤",
            f"│  Score  [{bar}] {s['best_score']:.3f}        │",
            f"│  Steps  {s['steps_taken']:>2}/{s['steps_taken'] + s['steps_remaining']:<2}   "
            f"Hints {s['hints_used']}   Bugs {s['n_bugs']}           │",
            "├" + "─" * 54 + "┤",
        ]

        for k, v in s["best_sub_scores"].items():
            sub_bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
            lines.append(f"│  {k:<14} [{sub_bar}] {v:.3f}          │")

        lines.append("└" + "─" * 54 + "┘")
        return "\n".join(lines)