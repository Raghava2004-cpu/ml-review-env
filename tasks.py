"""
tasks.py — The 3 ML bug tasks the agent must solve.
Easy → Medium → Hard
"""

TASKS = [

    # ============================================================
    # TASK 0 — EASY
    # Bugs: wrong loss reduction + missing zero_grad
    # ============================================================
    {
        "id": 0,
        "difficulty": "easy",
        "title": "Broken Training Loop",
        "description": (
            "This training loop has 2 bugs that cause silent training failure.\n"
            "Bug 1: Loss uses reduction='sum' instead of 'mean' — "
            "gradients are scaled by batch size, making learning rate unstable.\n"
            "Bug 2: optimizer.zero_grad() is missing — gradients accumulate "
            "across batches silently and corrupt every weight update.\n"
            "Fix both bugs and explain why each one causes silent failure."
        ),
        "buggy_code": """\
import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)

        # BUG 1: reduction='sum' inflates loss by batch size
        loss = criterion(outputs, targets, reduction='sum')

        # BUG 2: gradients from previous batch never cleared
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
""",
        "solution_code": """\
import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets, reduction='mean')

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
""",
        "bugs": [
            {
                "id": "wrong_reduction",
                "description": "Loss uses reduction='sum' instead of 'mean'",
                "keywords": ["mean", "reduction", "sum"],
                "severity": "high",
            },
            {
                "id": "missing_zero_grad",
                "description": "optimizer.zero_grad() missing before backward()",
                "keywords": ["zero_grad", "gradient", "clear"],
                "severity": "critical",
            },
        ],
        "architecture_checks": [
            "optimizer.zero_grad() called before loss.backward()",
            "loss uses reduction='mean'",
        ],
        "explanation_keywords": [
            "zero_grad", "gradient accumulation", "mean", "sum",
            "batch size", "learning rate", "converge", "silent"
        ],
        "hints": [
            "Hint 1: What happens if you never reset gradients between batches?",
            "Hint 2: If batch size is 32 and you use sum, your loss is 32x too large.",
            "Hint 3: Correct order is: zero_grad → forward → loss → backward → step",
        ],
    },

    # ============================================================
    # TASK 1 — MEDIUM
    # Bugs: no model.eval() + no torch.no_grad() + no model.train() restore
    # ============================================================
    {
        "id": 1,
        "difficulty": "medium",
        "title": "Broken Validation Loop",
        "description": (
            "This validation loop has 3 bugs that cause data leakage and memory waste.\n"
            "Bug 1: model.eval() never called — BatchNorm uses wrong statistics.\n"
            "Bug 2: torch.no_grad() missing — full computation graph stored in memory.\n"
            "Bug 3: model.train() never restored — next epoch trains with eval-mode BatchNorm.\n"
            "Fix all 3 bugs and explain the architectural flaw."
        ),
        "buggy_code": """\
import torch

def validate(model, dataloader, criterion):
    # BUG 1: model.eval() not called
    # BatchNorm uses per-batch stats instead of running stats

    val_loss = 0.0
    correct = 0
    total = 0

    # BUG 2: No torch.no_grad()
    # Full computation graph built for every batch — 2x memory waste
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    # BUG 3: model never set back to train() mode
    # Next training epoch runs with frozen BatchNorm stats

    return val_loss / len(dataloader), correct / total
""",
        "solution_code": """\
import torch

def validate(model, dataloader, criterion):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    model.train()

    return val_loss / len(dataloader), correct / total
""",
        "bugs": [
            {
                "id": "missing_eval_mode",
                "description": "model.eval() not called before validation",
                "keywords": ["eval", "batchnorm", "dropout", "running stats"],
                "severity": "critical",
            },
            {
                "id": "missing_no_grad",
                "description": "torch.no_grad() context manager missing",
                "keywords": ["no_grad", "computation graph", "memory"],
                "severity": "high",
            },
            {
                "id": "missing_train_restore",
                "description": "model.train() not called after validation",
                "keywords": ["train()", "restore", "mode"],
                "severity": "high",
            },
        ],
        "architecture_checks": [
            "model.eval() called at start of validation",
            "torch.no_grad() wraps the inference loop",
            "model.train() called after validation loop",
        ],
        "explanation_keywords": [
            "eval", "batchnorm", "dropout", "no_grad",
            "computation graph", "memory", "train mode", "running stats"
        ],
        "hints": [
            "Hint 1: PyTorch models have two modes. Which one should be active during validation?",
            "Hint 2: Every tensor operation builds a graph node. What prevents this during inference?",
            "Hint 3: Model state is global. If you change it in validate(), what state is it in next epoch?",
        ],
    },

    # ============================================================
    # TASK 2 — HARD
    # Bugs: wrong attention mask + wrong pos enc dim +
    #       broken weight tying + Pre/Post-LN mismatch
    # ============================================================
    {
        "id": 2,
        "difficulty": "hard",
        "title": "Broken Transformer Architecture",
        "description": (
            "This Transformer has 4 deep architectural bugs.\n"
            "Bug 1: Attention mask uses -1 instead of -inf — masked positions "
            "still receive ~9% attention weight after softmax.\n"
            "Bug 2: Positional encoding dim is 256 but d_model is 512 — "
            "shape mismatch crashes or silently broadcasts wrong values.\n"
            "Bug 3: Weight tying uses .data= which copies values once — "
            "embedding and output projection diverge after first update.\n"
            "Bug 4: Pre-LN and Post-LN styles are mixed — breaks gradient "
            "flow and pretrained weight compatibility.\n"
            "Fix all 4 bugs and explain each architectural decision."
        ),
        "buggy_code": """\
import torch
import torch.nn as nn

class BrokenTransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # BUG 4: Pre-LN applied (wrong for Post-LN transformer)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # BUG 1: mask uses -1 instead of -inf
        if mask is not None:
            mask = mask.float().masked_fill(mask == 0, -1)

        normed2 = self.norm2(x)
        ff_out = self.ff(normed2)
        x = x + self.dropout(ff_out)
        return x


class BrokenTransformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_layers=6,
                 n_heads=8, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        # BUG 2: dim is 256, should be d_model=512
        self.pos_encoding = nn.Embedding(max_seq_len, 256)

        self.layers = nn.ModuleList([
            BrokenTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # BUG 3: .data= copies values, doesn't tie the parameter
        self.output_proj.weight.data = self.embedding.weight.data

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output_proj(x)
""",
        "solution_code": """\
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.float().masked_fill(mask == 0, float('-inf'))

        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_layers=6,
                 n_heads=8, d_ff=2048, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output_proj(x)
""",
        "bugs": [
            {
                "id": "wrong_attention_mask",
                "description": "Attention mask uses -1 instead of -inf",
                "keywords": ["-inf", "float('-inf')", "softmax", "masked_fill"],
                "severity": "critical",
            },
            {
                "id": "wrong_pos_enc_dim",
                "description": "Positional encoding dim 256 != d_model 512",
                "keywords": ["d_model", "positional encoding", "dimension", "512"],
                "severity": "critical",
            },
            {
                "id": "broken_weight_tying",
                "description": "Weight tying uses .data= instead of direct assignment",
                "keywords": ["weight tying", ".weight =", "embedding", "parameter"],
                "severity": "high",
            },
            {
                "id": "pre_post_ln_mismatch",
                "description": "Pre-LN and Post-LN styles are mixed",
                "keywords": ["post-ln", "pre-ln", "residual", "layernorm"],
                "severity": "medium",
            },
        ],
        "architecture_checks": [
            "attention mask uses float('-inf') for padding",
            "pos_encoding uses d_model not hardcoded dim",
            "weight tying uses = not .data =",
            "layernorm applied post-residual consistently",
        ],
        "explanation_keywords": [
            "-inf", "softmax", "weight tying", "d_model",
            "positional encoding", "pre-ln", "post-ln", "residual", "GELU"
        ],
        "hints": [
            "Hint 1: softmax(-1) ≈ 0.09. softmax(-inf) = 0. Only one truly masks.",
            "Hint 2: Positional encodings are added to token embeddings element-wise. Dimensions must match.",
            "Hint 3: .data= copies values once. .weight= shares the same parameter object permanently.",
            "Hint 4: Pre-LN and Post-LN are not interchangeable. Pick one and use it consistently.",
        ],
    },
]