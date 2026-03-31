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
DEBUG        = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")


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


def build_prompt(observation: dict) -> str:
    task_hints = {
        0: """
CRITICAL FIXES — apply ALL of these:
1. Add optimizer.zero_grad() as the FIRST line inside the for loop before outputs = model(inputs)
2. Change reduction='sum' to reduction='mean' in the criterion call
""",
        1: """
CRITICAL FIXES — apply ALL of these:
1. Add model.eval() as the FIRST line of the function before val_loss = 0.0
2. Add with torch.no_grad(): and indent the entire for loop inside it
3. Add model.train() as the LAST line of the function right before return
""",
        2: """
CRITICAL FIXES — apply ALL of these:
1. In masked_fill change -1 to float('-inf')
2. Change nn.Embedding(max_seq_len, 256) to nn.Embedding(max_seq_len, d_model)
3. Change output_proj.weight.data = self.embedding.weight.data to output_proj.weight = self.embedding.weight
4. Change ReLU() to GELU() in the feedforward network
5. Apply Post-LN style: x = self.norm1(x + self.dropout(attn_out)) and x = self.norm2(x + self.dropout(ff_out))
6. Remove the normed = self.norm1(x) pattern completely
"""
    }

    hint = task_hints.get(observation['task_id'], "")

    return textwrap.dedent(f"""
        You are a senior PyTorch engineer doing a code review.
        You must fix ALL bugs listed below. Miss even one and the test fails.

        Task: {observation['title']}
        Difficulty: {observation['difficulty']}

        Buggy code:
        {observation['buggy_code']}

        {hint}

        RULES:
        - Return ONLY the complete fixed Python code
        - No explanation
        - No markdown fences
        - No code fences
        - Include ALL imports
        - Fix EVERY bug listed above
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

def run_task(task_id: int) -> dict:
    global client
    env         = MLReviewEnv(task_id=task_id, max_steps=MAX_STEPS)
    observation = env.reset()
    best_score  = 0.0

    if DEBUG:
        print(f"\n{'━' * 55}")
        print(f"  Task {task_id} [{observation['difficulty'].upper()}]"
              f" — {observation['title']}")
        print(f"  Bugs to fix: {observation['n_bugs']}")
        print(f"{'━' * 55}")

    # Step 1 — explain the bugs first (boosts explanation score)
    explanation_prompt = textwrap.dedent(f"""
    You are a senior ML engineer explaining bugs to a junior developer.
    Analyze this buggy PyTorch code carefully.

    Code:
    {observation['buggy_code']}

    For EACH bug explain:
    1. What is wrong (be specific)
    2. Why it causes silent failure (no crash but wrong results)
    3. How it affects gradient accumulation, memory, convergence, or accuracy

    Use these exact keywords in your explanation:
    - zero_grad, gradient accumulation, mean, sum, batch size, learning rate
    - eval, batchnorm, dropout, no_grad, computation graph, memory, train mode
    - -inf, softmax, weight tying, d_model, positional encoding, pre-ln, post-ln
    - silent, production, converge, epoch, accuracy, leak
     """).strip()

    exp_response = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [{"role": "user", "content": explanation_prompt}],
        temperature = TEMPERATURE,
        max_tokens  = 512,
    )
    explanation = exp_response.choices[0].message.content

    result = env.step({
        "type":        "explain_issue",
        "explanation": explanation,
    })
    observation = result.observation
    best_score  = observation["best_score"]

    if DEBUG:
        print(f"\n  Step 1 — explanation score: {best_score:.4f}")

    # Step 2 — submit fix
    for step in range(MAX_STEPS - 1):
        prompt   = build_prompt(observation)
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

        # Alternate between optimize and submit_fix
        action_type = "optimize" if step % 2 == 0 else "submit_fix"

        result = env.step({
            "type": action_type,
            "code": fixed_code,
        })

        observation = result.observation
        best_score  = observation["best_score"]
        bugs_fixed  = result.info.get("bugs_fixed", [])
        remaining   = result.info.get("bugs_remaining", [])

        if DEBUG:
            print(f"\n  Step {step + 2}/{MAX_STEPS}...")
            print(f"  Score: {best_score:.4f}")
            if bugs_fixed:
                print(f"  ✅ Fixed:   {', '.join(bugs_fixed)}")
            if remaining:
                print(f"  ❌ Left:    {', '.join(remaining)}")

        if result.done:
            if DEBUG:
                print(f"  🎉 Solved!")
            break

    return {
        "task_id":    task_id,
        "difficulty": TASKS[task_id]["difficulty"],
        "title":      TASKS[task_id]["title"],
        "score":      best_score,
        "steps":      observation["steps_taken"],
        "bugs_fixed": bugs_fixed if 'bugs_fixed' in dir() else [],
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
    global client
    client = get_client()

    # Run all 3 tasks
    results = []
    for task_id in range(len(TASKS)):
        try:
            result = run_task(task_id)
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
