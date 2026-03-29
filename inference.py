"""
inference.py — Inference script for MLReviewEnv
Required by hackathon spec. Must be named inference.py.

Reads from environment variables:
    API_BASE_URL  — LLM API endpoint
    MODEL_NAME    — model identifier
    HF_TOKEN      — Hugging Face / API key

Uses OpenAI client for all LLM calls.
Runtime: under 20 minutes on vcpu=2, memory=8gb
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import textwrap
from openai import OpenAI
from env import MLReviewEnv
from tasks import TASKS


# ─────────────────────────────────────────────────────────────
# Read environment variables
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MAX_STEPS    = 8
TEMPERATURE  = 0.2
MAX_TOKENS   = 2048


# ─────────────────────────────────────────────────────────────
# Validate env vars are set
# ─────────────────────────────────────────────────────────────

def validate_env():
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        print(f"\n❌ ERROR: Missing environment variables: {', '.join(missing)}")
        print("   Set them in your .env file:")
        print("   API_BASE_URL=https://router.huggingface.co/v1")
        print("   MODEL_NAME=Qwen/Qwen2.5-72B-Instruct")
        print("   HF_TOKEN=hf_your_token_here\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Build OpenAI client using env variables
# ─────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    return OpenAI(
        api_key  = HF_TOKEN,
        base_url = API_BASE_URL,
    )


# ─────────────────────────────────────────────────────────────
# Build prompt for the LLM
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior ML engineer specializing in PyTorch training code.
    You find and fix bugs in ML training loops, validation loops,
    and neural network architectures.
    When given buggy code, you return ONLY the fixed Python code.
    No explanation. No markdown fences. Just raw Python code.
""").strip()


def build_prompt(state: dict) -> str:
    return textwrap.dedent(f"""
        Task: {state['title']}
        Difficulty: {state['difficulty']}

        Description:
        {state['description']}

        Buggy code to fix:
        {state['buggy_code']}

        Return ONLY the complete fixed Python code.
        No explanation. No markdown. No code fences.
        Just the raw fixed Python code.
    """).strip()


# ─────────────────────────────────────────────────────────────
# Clean LLM response
# ─────────────────────────────────────────────────────────────

def clean_code(raw: str) -> str:
    code = raw.strip()
    # Remove markdown fences if model added them
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


# ─────────────────────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: int) -> dict:
    env   = MLReviewEnv(task_id=task_id, max_steps=MAX_STEPS)
    state = env.reset()

    print(f"\n{'━' * 55}")
    print(f"  Task {task_id} [{state['difficulty'].upper()}] — {state['title']}")
    print(f"  Bugs to fix: {state['n_bugs']}")
    print(f"{'━' * 55}")

    best_score = 0.0
    bugs_fixed = []

    for step in range(MAX_STEPS):
        print(f"\n  Step {step + 1}/{MAX_STEPS}...")

        # Build prompt from current state
        prompt = build_prompt(state)

        # Call LLM using OpenAI client
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )

        fixed_code = clean_code(response.choices[0].message.content)

        # Submit fix to environment
        result = env.step({
            "type": "submit_fix",
            "code": fixed_code,
        })

        # Update state for next step
        state      = result.observation
        best_score = state["best_score"]
        bugs_fixed = result.info.get("bugs_fixed", [])
        remaining  = result.info.get("bugs_remaining", [])

        print(f"  Score: {best_score:.4f}")
        if bugs_fixed:
            print(f"  ✅ Fixed:     {', '.join(bugs_fixed)}")
        if remaining:
            print(f"  ❌ Remaining: {', '.join(remaining)}")

        # Stop early if all bugs fixed
        if result.done:
            print(f"  🎉 Solved in {step + 1} steps!")
            break

    return {
        "task_id":        task_id,
        "difficulty":     TASKS[task_id]["difficulty"],
        "title":          TASKS[task_id]["title"],
        "score":          best_score,
        "steps":          state["steps_taken"],
        "bugs_fixed":     bugs_fixed,
    }


# ─────────────────────────────────────────────────────────────
# Print final results table
# ─────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    W = 65
    print(f"\n{'═' * W}")
    print(f"  BASELINE SCORES — MLReviewEnv")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  API URL:  {API_BASE_URL}")
    print(f"{'═' * W}")
    print(f"  {'Task':<4}  {'Difficulty':<10}  {'Title':<25}  {'Score':>6}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*25}  {'─'*6}")

    total = 0.0
    for r in results:
        print(
            f"  {r['task_id']:<4}  "
            f"{r['difficulty']:<10}  "
            f"{r['title'][:25]:<25}  "
            f"{r['score']:>6.4f}"
        )
        total += r["score"]

    avg = total / len(results) if results else 0.0
    print(f"{'═' * W}")
    print(f"  Average score: {avg:.4f}")
    print(f"{'═' * W}\n")

    return avg


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def main():
    print("\n🚀 MLReviewEnv — Inference Script")
    print(f"   API:   {API_BASE_URL}")
    print(f"   Model: {MODEL_NAME}")

    # Validate env vars
    if not validate_env():
        return

    # Create OpenAI client
    client = get_client()

    # Run all 3 tasks
    results = []
    for task_id in range(len(TASKS)):
        try:
            result = run_task(client, task_id)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Task {task_id} failed: {e}")
            results.append({
                "task_id":    task_id,
                "difficulty": TASKS[task_id]["difficulty"],
                "title":      TASKS[task_id]["title"],
                "score":      0.0,
                "steps":      0,
                "bugs_fixed": [],
            })

    # Print final table
    avg = print_results(results)

    # Return results for programmatic use
    return {
        "model":   MODEL_NAME,
        "average": avg,
        "tasks":   results,
    }


if __name__ == "__main__":
    main()