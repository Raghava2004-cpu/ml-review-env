---
title: ML Code Review Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
tags:
  - openenv
---

# MLReviewEnv — ML Training Code Review Environment

An OpenEnv-compatible environment where an AI agent reviews and fixes
senior-level ML training code bugs. Tasks range from broken training
loops to deep Transformer architecture flaws.

---

## Why This Environment?

Silent bugs in ML training code are the hardest to catch:
- No crash, no error — the model just silently fails to converge
- Requires deep PyTorch knowledge to spot
- Real bugs that cost teams days of debugging in production

This environment trains agents to catch exactly these bugs.

---

## Tasks

| ID | Difficulty | Title | Bugs |
|----|-----------|-------|------|
| 0 | Easy | Broken Training Loop | 2 |
| 1 | Medium | Broken Validation Loop | 3 |
| 2 | Hard | Broken Transformer Architecture | 4 |

### Task 0 — Easy: Broken Training Loop
The training loop has 2 bugs that cause silent training failure:
- **Bug 1:** `reduction='sum'` inflates loss by batch size — effective learning rate is unstable
- **Bug 2:** `optimizer.zero_grad()` missing — gradients accumulate across batches silently

### Task 1 — Medium: Broken Validation Loop
The validation loop has 3 bugs causing data leakage and memory waste:
- **Bug 1:** `model.eval()` never called — BatchNorm uses wrong per-batch statistics
- **Bug 2:** `torch.no_grad()` missing — full computation graph stored, 2x memory waste
- **Bug 3:** `model.train()` never restored — next epoch trains with frozen BatchNorm stats

### Task 2 — Hard: Broken Transformer Architecture
4 deep architectural bugs that are nearly impossible to spot without expertise:
- **Bug 1:** Attention mask uses `-1` instead of `-inf` — masked positions still get ~9% attention
- **Bug 2:** Positional encoding dim `256` != `d_model 512` — shape mismatch
- **Bug 3:** Weight tying uses `.data=` — embedding and output projection diverge after first update
- **Bug 4:** Pre-LN and Post-LN mixed — breaks gradient flow and pretrained weight compatibility

---

## Action Space

The agent can take 5 types of actions each step:

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `submit_fix` | `code: str` | Submit corrected Python code |
| `explain_issue` | `explanation: str` | Explain what the bugs are and why |
| `optimize` | `code: str` | Submit code with performance improvements |
| `request_hint` | — | Get a hint (costs -0.04 reward) |
| `no_op` | — | Do nothing (costs -0.02 reward) |

---

## Observation Space

What the agent sees after every step:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | int | 0=easy, 1=medium, 2=hard |
| `difficulty` | str | easy / medium / hard |
| `title` | str | Short task title |
| `description` | str | Full task description with bug hints |
| `buggy_code` | str | The broken ML code to fix |
| `n_bugs` | int | Number of bugs in this task |
| `steps_taken` | int | Steps used so far |
| `steps_remaining` | int | Steps left before episode ends |
| `hints_used` | int | Number of hints requested |
| `best_score` | float | Best reward achieved this episode |
| `best_sub_scores` | dict | Best score per dimension |
| `done` | bool | Is the episode finished? |

---

## Reward Function

Total reward is always between **-0.2 and 1.0**:

```
reward = 0.45 × correctness     (are bugs fixed?)
       + 0.20 × architecture    (is design pattern correct?)
       + 0.20 × explanation     (does agent explain WHY?)
       + 0.10 × efficiency      (no memory waste?)
       + 0.05 × speed bonus     (solved with steps remaining?)
       - 0.04 × hints used      (penalty per hint)
```

**Partial progress:** Every individual bug fix contributes to the
correctness score — the agent gets reward signal even for fixing
1 out of 3 bugs.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/tasks` | List all tasks + action schema |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take one action |
| GET | `/state` | Get current state |
| POST | `/grader` | Score a submission |
| GET | `/baseline` | Run baseline LLM agent |

---

## Setup & Usage

### Run locally

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/open-env
cd ml-review-env

# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py
```

Server runs at `http://localhost:7860`

### Run with Docker

```bash
docker build -t ml-review-env .
docker run -p 7860:7860 ml-review-env
```

### Run baseline

```bash
# Set your OpenAI API key
set OPENAI_API_KEY=sk-your-key-here      # Windows
export OPENAI_API_KEY=sk-your-key-here   # Mac/Linux

# Run baseline on all tasks
python baseline.py --verbose

# Run only one task
python baseline.py --task 0
```

---

## Baseline Scores

Scores achieved by `gpt-4o-mini` on each task:

| Task | Difficulty | Score |
|------|-----------|-------|
| 0 | Easy | 0.6967 |
| 1 | Medium | 0.6967 |
| 2 | Hard | 0.6500 |
| **Average** | | **0.6811** |

---

## Project Structure

```
ml-review-env/
  ├── env.py           # Core environment: reset/step/state API
  ├── tasks.py         # 3 ML bug task definitions
  ├── graders.py       # Scores agent answers (AST + pattern analysis)
  ├── models.py        # Pydantic typed models (OpenEnv spec)
  ├── server.py        # FastAPI HTTP server (6 endpoints)
  ├── baseline.py      # OpenAI-powered baseline agent
  ├── openenv.yaml     # OpenEnv spec metadata
  ├── Dockerfile       # Container for HF Spaces deployment
  ├── requirements.txt # Python dependencies
  └── README.md        # This file
```
