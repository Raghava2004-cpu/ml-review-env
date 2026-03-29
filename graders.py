"""
graders.py — Scores the agent's submitted code against each task.
Returns sub-scores that feed the reward function.
All scores are in [0.0, 1.0]
"""

import ast
import re
from typing import Any


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def grade(task: dict, action: dict) -> dict:
    """
    Grade any action against the given task.
    action = {
        "type": "submit_fix" | "explain_issue" | "optimize" | "request_hint" | "no_op"
        "code": str         (for submit_fix / optimize)
        "explanation": str  (for explain_issue)
    }
    Returns dict with correctness_score, architecture_score,
    explanation_score, efficiency_score, bugs_fixed, bugs_remaining.
    """
    action_type = action.get("type", "no_op")

    if action_type == "submit_fix":
        return _grade_fix(task, action.get("code", ""))

    elif action_type == "explain_issue":
        return _grade_explanation(task, action.get("explanation", ""))

    elif action_type == "optimize":
        result = _grade_fix(task, action.get("code", ""))
        result["efficiency_score"] = _grade_efficiency(task, action.get("code", ""))
        return result

    else:
        # no_op or request_hint — no score
        return _zero(task)


# ─────────────────────────────────────────────────────────────
# Fix grader
# ─────────────────────────────────────────────────────────────

def _grade_fix(task: dict, code: str) -> dict:
    details: dict[str, Any] = {}

    # Step 1: can it even be parsed?
    try:
        ast.parse(code)
        details["parses"] = True
    except SyntaxError as e:
        details["parses"] = False
        details["syntax_error"] = str(e)
        return _zero(task, details)

    bugs = task["bugs"]
    checks = task.get("architecture_checks", [])

    # Step 2: check each bug individually
    fixed = []
    remaining = []
    bug_details = {}

    for bug in bugs:
        is_fixed, evidence = _check_bug_fixed(bug, code)
        bug_details[bug["id"]] = {"fixed": is_fixed, "evidence": evidence}
        if is_fixed:
            fixed.append(bug["id"])
        else:
            remaining.append(bug["id"])

    n_bugs = len(bugs)
    n_fixed = len(fixed)
    correctness = round(n_fixed / n_bugs, 4) if n_bugs else 0.0
    details["bug_details"] = bug_details

    # Step 3: architecture pattern checks
    arch_passed = [c for c in checks if _check_arch(c, code)]
    arch_failed  = [c for c in checks if not _check_arch(c, code)]
    architecture = round(len(arch_passed) / len(checks), 4) if checks else 1.0
    details["arch_passed"] = arch_passed
    details["arch_failed"] = arch_failed

    return {
        "correctness_score":  correctness,
        "architecture_score": architecture,
        "explanation_score":  0.0,
        "efficiency_score":   0.0,
        "partial_score":      correctness,
        "bugs_fixed":         fixed,
        "bugs_remaining":     remaining,
        "details":            details,
    }


# ─────────────────────────────────────────────────────────────
# Per-bug detection  (AST + regex patterns)
# ─────────────────────────────────────────────────────────────

def _check_bug_fixed(bug: dict, code: str) -> tuple[bool, str]:
    bid = bug["id"]

    # ── Task 0 ───────────────────────────────────────────────
    if bid == "missing_zero_grad":
        lines = code.splitlines()
        zg = next((i for i, l in enumerate(lines) if "zero_grad" in l), None)
        bw = next((i for i, l in enumerate(lines) if ".backward" in l), None)
        if zg is not None and bw is not None and zg < bw:
            return True, "zero_grad() found before backward()"
        if zg is not None:
            return False, "zero_grad() found but NOT before backward()"
        return False, "zero_grad() not found at all"

    if bid == "wrong_reduction":
        has_sum  = bool(re.search(r"reduction\s*=\s*['\"]sum['\"]", code))
        has_mean = bool(re.search(r"reduction\s*=\s*['\"]mean['\"]", code))
        if has_sum:
            return False, "reduction='sum' still present"
        if has_mean or "reduction" not in code:
            return True, "reduction='mean' used or default (mean)"
        return False, "reduction setting unclear"

    # ── Task 1 ───────────────────────────────────────────────
    if bid == "missing_eval_mode":
        found = bool(re.search(r"model\s*\.\s*eval\s*\(\s*\)", code))
        return found, "model.eval() found" if found else "model.eval() missing"

    if bid == "missing_no_grad":
        as_context = bool(re.search(r"with\s+torch\.no_grad", code))
        if as_context:
            return True, "torch.no_grad() used as context manager"
        if "no_grad" in code:
            return False, "no_grad referenced but not used as context manager"
        return False, "torch.no_grad() not found"

    if bid == "missing_train_restore":
        lines = code.splitlines()
        last_for = max(
            (i for i, l in enumerate(lines) if l.strip().startswith("for ")),
            default=-1
        )
        train_lines = [
            i for i, l in enumerate(lines)
            if re.search(r"model\s*\.\s*train\s*\(", l)
        ]
        if train_lines and max(train_lines) > last_for:
            return True, "model.train() found after validation loop"
        if train_lines:
            return False, "model.train() found but BEFORE the loop"
        return False, "model.train() not found"

    # ── Task 2 ───────────────────────────────────────────────
    if bid == "wrong_attention_mask":
        uses_neg_inf = (
            "float('-inf')" in code or
            'float("-inf")' in code or
            bool(re.search(r"-1e[0-9]", code))
        )
        still_minus_one = bool(re.search(r"masked_fill\s*\(.*,\s*-1\s*\)", code))
        if uses_neg_inf and not still_minus_one:
            return True, "Using -inf for attention mask (correct)"
        if still_minus_one:
            return False, "Still using -1 for mask (should be -inf)"
        return False, "No mask fill value found"

    if bid == "wrong_pos_enc_dim":
        wrong = bool(re.search(r"Embedding\s*\([^)]*,\s*256\s*\)", code))
        right = bool(re.search(r"Embedding\s*\([^)]*,\s*d_model\s*\)", code))
        if wrong:
            return False, "pos_encoding still uses dim=256"
        if right:
            return True, "pos_encoding uses d_model (correct)"
        right_val = bool(re.search(r"Embedding\s*\([^)]*,\s*512\s*\)", code))
        return right_val, ("pos_encoding uses 512" if right_val else "Dim unclear")

    if bid == "broken_weight_tying":
        wrong_tie = bool(re.search(r"weight\.data\s*=", code))
        right_tie = bool(re.search(
            r"output_proj\.weight\s*=\s*self\.embedding\.weight(?!\.data)", code
        ))
        if wrong_tie:
            return False, "Still using .data= (value copy, not shared parameter)"
        if right_tie:
            return True, "Direct parameter assignment — correct weight tying"
        return False, "Weight tying not found"

    if bid == "pre_post_ln_mismatch":
        post_ln = bool(re.search(r"norm\d*\s*\(\s*x\s*\+", code))
        pre_ln  = bool(re.search(r"normed\s*=\s*self\.norm\d*\s*\(\s*x\s*\)", code))
        if post_ln and not pre_ln:
            return True, "Post-LN applied consistently (correct)"
        if pre_ln:
            return False, "Pre-LN pattern still detected"
        return False, "LayerNorm placement unclear"

    # Fallback: keyword matching
    keywords = bug.get("keywords", [])
    hits = sum(1 for kw in keywords if kw.lower() in code.lower())
    score = hits / len(keywords) if keywords else 0
    return score > 0.5, f"Keyword match: {hits}/{len(keywords)}"


# ─────────────────────────────────────────────────────────────
# Architecture checks
# ─────────────────────────────────────────────────────────────

def _check_arch(check: str, code: str) -> bool:
    c = check.lower()

    if "zero_grad" in c and "before" in c:
        lines = code.splitlines()
        zg = next((i for i, l in enumerate(lines) if "zero_grad" in l), None)
        bw = next((i for i, l in enumerate(lines) if ".backward" in l), None)
        return zg is not None and bw is not None and zg < bw

    if "mean" in c and "reduction" in c:
        return not bool(re.search(r"reduction\s*=\s*['\"]sum['\"]", code))

    if "model.eval()" in c:
        return bool(re.search(r"model\s*\.\s*eval\s*\(\)", code))

    if "no_grad" in c:
        return bool(re.search(r"with\s+torch\.no_grad", code))

    if "model.train()" in c:
        return bool(re.search(r"model\s*\.\s*train\s*\(", code))

    if "-inf" in c or "float" in c:
        return (
            "float('-inf')" in code or
            'float("-inf")' in code or
            bool(re.search(r"-1e[0-9]", code))
        )

    if "d_model" in c and "pos" in c:
        return not bool(re.search(r"Embedding\s*\([^)]*,\s*256\s*\)", code))

    if "weight" in c and "tying" in c:
        return not bool(re.search(r"weight\.data\s*=", code))

    if "post-residual" in c or "post" in c and "ln" in c:
        return bool(re.search(r"norm\d*\s*\(\s*x\s*\+", code))

    return any(kw in code for kw in check.split())


# ─────────────────────────────────────────────────────────────
# Explanation grader
# ─────────────────────────────────────────────────────────────

def _grade_explanation(task: dict, explanation: str) -> dict:
    if not explanation.strip():
        return _zero(task, {"explanation": "empty"})

    keywords = task.get("explanation_keywords", [])
    bugs     = task["bugs"]
    exp_low  = explanation.lower()

    # keyword coverage
    kw_hits  = [kw for kw in keywords if kw.lower() in exp_low]
    kw_score = len(kw_hits) / len(keywords) if keywords else 0.0

    # per-bug coverage
    bug_scores = []
    for bug in bugs:
        bkw  = bug.get("keywords", [])
        hits = sum(1 for kw in bkw if kw.lower() in exp_low)
        bug_scores.append(hits / len(bkw) if bkw else 0.5)
    per_bug = sum(bug_scores) / len(bug_scores) if bug_scores else 0.0

    # depth bonus (mentions real-world impact)
    depth_kw = ["silent", "production", "converge", "memory",
                 "gradient", "leak", "epoch", "accuracy"]
    depth    = min(0.2, sum(1 for kw in depth_kw if kw in exp_low) * 0.05)

    score = min(1.0, 0.5 * kw_score + 0.4 * per_bug + depth)

    return {
        "correctness_score":  0.0,
        "architecture_score": 0.0,
        "explanation_score":  round(score, 4),
        "efficiency_score":   0.0,
        "partial_score":      round(score * 0.5, 4),
        "bugs_fixed":         [],
        "bugs_remaining":     [b["id"] for b in bugs],
        "details":            {"keywords_hit": kw_hits},
    }


# ─────────────────────────────────────────────────────────────
# Efficiency grader
# ─────────────────────────────────────────────────────────────

def _grade_efficiency(task: dict, code: str) -> float:
    checks = {
        "no_grad_used":    bool(re.search(r"with\s+torch\.no_grad", code)),
        "eval_mode":       bool(re.search(r"model\s*\.\s*eval\s*\(\)", code)),
        "no_data_copy":    ".data =" not in code,
        "uses_gelu":       "GELU" in code or "gelu" in code.lower(),
    }
    passed = sum(checks.values())
    return round(passed / len(checks), 4)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _zero(task: dict, details: dict = {}) -> dict:
    return {
        "correctness_score":  0.0,
        "architecture_score": 0.0,
        "explanation_score":  0.0,
        "efficiency_score":   0.0,
        "partial_score":      0.0,
        "bugs_fixed":         [],
        "bugs_remaining":     [b["id"] for b in task["bugs"]],
        "details":            details,
    }