"""
buggy_transformer.py -- the flagship debugging drill (OpenAI-style ML round).

THIS FILE CONTAINS 7 PLANTED BUGS. The model trains without crashing --
every shape works out and no exception is raised -- but it learns poorly.
Your job:

    1. Find and fix all 7 bugs.                     (time box: 45 minutes)
    2. Extension: implement KV-cached generation.   (+30 minutes)

Rules of engagement (this mirrors the real round):
  * Run the file FIRST. Look at the loss curve and the generated samples
    before reading a single line of model code. A healthy run of this
    exact architecture and budget reaches train loss well under 2.0 by
    the last step and emits English-ish text. Note what you actually get.
  * Narrate as you go: state a hypothesis, run a check, accept or reject.
    Silent staring is what fails this round, not slow typing.
  * Do not open buggy_transformer_solutions.md until done or out of time.

The intended architecture is a standard pre-LN decoder-only transformer:

    tokens -> tok_emb + pos_emb -> [Block x 2] -> LayerNorm -> Linear -> logits
    Block:   x = x + attn(LN(x));  x = x + mlp(LN(x))

Run: uv run python ml/08_interview_drills/buggy_transformer.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# 0. The corpus, embedded here -- original prose written for this file,
#    nothing downloaded. Character-level modeling needs very little text
#    to learn spelling, spacing, and short-word statistics.
# ---------------------------------------------------------------------------
CORPUS = """
The keeper of the lamp at Graywater Point had one rule, and the rule was
this. The light turns whether or not you believe in ships. He said it to
the gulls, to the kettle, and to every apprentice the mainland sent him,
and the apprentices wrote it down because he said nothing else for days
at a stretch.

One apprentice stayed longer than the rest. She learned to trim the wick
by feel, to read the weather in the ache of the iron rail, and to sleep
in the afternoon so the night could belong to the light. She asked him
once why the rule was the rule. He rolled a coin of wax between his
fingers and said that a ship you never see is still a ship, and a rock
you never hit is still a rock, and the sea does not grade you on your
reasons.

In her third winter a storm took out the glass on the landward side, and
they worked to dawn with rope and canvas while the flame leaned and
gasped and held. In the morning the sea was flat and bright and empty,
and it looked as if nothing at all had happened, and that, he said, was
the whole wage of the work. Nothing at all had happened. Nothing at all
would keep happening, night after night, as long as somebody kept the
lamp.

When he retired to the mainland he left her the log books, forty years
of small hours in a careful hand. Nearly every page said the same thing.
Lamp lit. Sea heavy or sea calm. No wrecks. She read them like scripture
all the same, for she knew now what the pages were counting, and it was
not nothing. It was every ship that sailed past Graywater in the dark
and never learned their names.
"""

# ---------------------------------------------------------------------------
# 1. Tokenizer: character-level.
#    encode: str -> list[int], decode: list[int] -> str.
# ---------------------------------------------------------------------------
chars = sorted(set(CORPUS))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # char -> int
itos = {i: ch for ch, i in stoi.items()}       # int -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)

data = torch.tensor(encode(CORPUS), dtype=torch.long)   # (len_corpus,)
print(f"corpus: {len(CORPUS)} chars, vocab_size = {vocab_size}")

# ---------------------------------------------------------------------------
# 2. Hyperparameters -- tiny on purpose; a full run is ~1 minute on CPU.
#    These values are NOT the problem. All 7 bugs are in code, not config.
# ---------------------------------------------------------------------------
block_size = 64    # max context length T
n_embd = 64        # embedding width d_model
n_head = 4         # attention heads; head_dim = 64 / 4 = 16
n_layer = 2        # transformer blocks
batch_size = 32
max_steps = 300
lr = 3e-3


def get_batch():
    """Sample batch_size random crops.
    x: (B, T) input chars; y: (B, T) = x shifted left by one -- every
    position t is a training example: predict char t+1 from chars <= t."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T)
    return x, y


# ---------------------------------------------------------------------------
# 3. Multi-head causal self-attention, written by hand.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = n_embd // n_head
        # One fused linear computes Q, K, V for all heads at once:
        # (B, T, n_embd) -> (B, T, 3*n_embd), then split three ways.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)   # output mixing W_o
        # Causal mask so a position cannot attend to the future.
        # register_buffer = part of the module, but not a parameter.
        self.register_buffer(
            "mask", torch.triu(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape                       # C = n_embd
        qkv = self.qkv(x)                       # (B, T, 3C)
        q, k, v = qkv.split(n_embd, dim=2)      # each (B, T, C)
        # Split channels into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_dim).transpose(1, 2)

        # Attention scores: (B,h,T,d) @ (B,h,d,T) -> (B,h,T,T),
        # scaled so the scores keep roughly unit variance at init.
        att = (q @ k.transpose(-2, -1)) / n_embd
        # Masked positions get -inf, which softmax turns into weight 0.
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        # Normalize scores into attention weights.
        att = F.softmax(att, dim=-2)            # (B, h, T, T)
        y = att @ v                             # (B,h,T,T)@(B,h,T,d) -> (B,h,T,d)
        # Merge heads back: (B, h, T, d) -> (B, T, h, d) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, nonlinearity, project back.
    Attention moves information BETWEEN tokens; this computes per token."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)    # (C)  -> (4C)
        self.proj = nn.Linear(4 * n_embd, n_embd)  # (4C) -> (C)

    def forward(self, x):                          # (B, T, C) -> (B, T, C)
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-LN transformer block: normalize, transform, add residual."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):                          # (B, T, C)
        h = self.ln1(x)
        x = h + self.attn(h)                       # communicate
        x = x + self.mlp(self.ln2(x))              # compute
        return x


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # (V, C) lookup
        self.pos_emb = nn.Embedding(block_size, n_embd)   # (T_max, C) learned
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)         # (C) -> (V) logits

    def forward(self, idx, targets=None):
        # idx: (B, T) integer char ids
        B, T = idx.shape
        tok = self.tok_emb(idx)                            # (B, T, C)
        pos = self.pos_emb(idx)                            # (B, T, C)
        x = tok + pos
        for block in self.blocks:
            x = block(x)                                   # (B, T, C)
        x = self.ln_f(x)                                   # (B, T, C)
        logits = self.head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            # Cross-entropy over all B*T positions at once: V-way
            # classification, "which character comes next?"
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size), targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx: (B, T0) prompt. Autoregressive: sample, append, repeat.
        # (Extension: this recomputes all T positions every step --
        #  rewrite it with a KV cache once the bugs are fixed.)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]           # crop to window
            logits, _ = self(idx_cond)                # (B, T, V)
            logits = logits[:, -1, :] / temperature   # (B, V) last position
            probs = F.softmax(logits, dim=-1)         # (B, V)
            nxt = torch.multinomial(probs, 1)         # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)        # (B, T+1)
        return idx


# ---------------------------------------------------------------------------
# 4. Train.
# ---------------------------------------------------------------------------
def sample_text(model, n=200):
    start = torch.zeros((1, 1), dtype=torch.long)   # token 0 is '\n'
    out = model.generate(start, n, temperature=0.8)
    return decode(out[0].tolist())


def main():
    model = TinyLM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    # A model that predicts uniformly over the vocab has loss ln(V).
    # That is the "knows nothing" baseline -- keep it in view.
    print(f"ln(vocab_size) = {math.log(vocab_size):.3f}  "
          f"<- loss of a model that has learned nothing\n")

    print("--- sample BEFORE training ---")
    print(sample_text(model))
    print("------------------------------\n")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(max_steps):
        x, y = get_batch()                 # (B, T), (B, T)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == max_steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")
    dt = time.time() - t0
    print(f"\ntrained {max_steps} steps in {dt:.1f}s "
          f"({dt / max_steps * 1000:.0f} ms/step)")

    print("\n--- sample AFTER training ---")
    print(sample_text(model))
    print("-----------------------------")
    print("\nA correct implementation ends well under loss 2.0 on this "
          "budget\nand the sample reads like the corpus. If yours does "
          "not, the bugs\nare still in there. Find all 7.")


if __name__ == "__main__":
    main()
