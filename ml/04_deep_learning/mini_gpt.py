"""
mini_gpt.py -- a minimal but complete GPT, character-level, CPU, ~1 minute.

In the spirit of nanoGPT: everything explicit, every tensor shape annotated.
The whole model is:

    tokens -> tok_emb + pos_emb -> [Block x 2] -> LayerNorm -> Linear -> logits

where Block = x + attn(LN(x))  then  x + mlp(LN(x))   (pre-LN + residuals),
and attn is multi-head CAUSAL self-attention written by hand (no
nn.Transformer, no nn.MultiheadAttention).

Training objective: next-character prediction (cross-entropy). We print a
text sample BEFORE training (pure noise) and AFTER (recognizable English-ish
character statistics) to make "the loss going down" mean something.

Run: uv run python ml/04_deep_learning/mini_gpt.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 0. The corpus, embedded right here. A few paragraphs of original
#    fable-flavored prose (written for this file; nothing downloaded).
#    Char-level modeling needs surprisingly little text to learn spelling,
#    spacing, and short-word statistics.
# ---------------------------------------------------------------------------
CORPUS = """
The old miller kept a small grey cat, and the cat kept the mill. Every
morning the miller ground his grain, and every morning the cat walked the
beams above him, watching the wheat dust drift like slow snow. The miller
said the cat earned no bread, for it did no work. The cat said nothing at
all, for it was a cat, and the mice said nothing either, for they had long
ago moved to the baker's house across the river.

One winter the river froze and the wheel stood still. The miller counted
his sacks and found them full, though he had not ground grain in a month.
He counted again by candlelight, and a third time at dawn. The sacks were
full because nothing had touched them, and nothing had touched them because
the small grey cat still walked the beams, silent as the snow on the roof.

The miller was not a quick man, but he was not a fool either. He set a
bowl of warm milk by the door and never again spoke of bread, or of work,
or of what the cat had earned. Some wages are paid in what does not happen.
The wheel turned in spring, the grain flowed, and the cat grew old and
grey and fat, and slept in the sun like a stone that had decided to be
soft.

A young man from the town once asked the miller the secret of his tidy
mill. The miller pointed at the sleeping cat. The young man laughed and
said a sleeping cat does nothing. The miller nodded and poured the milk.
It is hard to teach a man the value of nothing happening, and the miller
did not try. He ground his grain. The wheel turned. The cat slept on, and
the mice stayed across the river, and that was the whole of the secret,
and it was enough.
"""

# ---------------------------------------------------------------------------
# 1. Tokenizer: character-level. vocab = sorted unique chars in the corpus.
#    encode: str -> list[int], decode: list[int] -> str. That's all a
#    tokenizer is; BPE just picks smarter units than single characters.
# ---------------------------------------------------------------------------
chars = sorted(set(CORPUS))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # char -> int
itos = {i: ch for ch, i in stoi.items()}       # int -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[int(i)] for i in ids)

data = torch.tensor(encode(CORPUS), dtype=torch.long)   # (len_corpus,)
print(f"corpus: {len(CORPUS)} chars, vocab_size = {vocab_size}")
print(f"vocab: {''.join(chars)!r}")

# ---------------------------------------------------------------------------
# 2. Hyperparameters -- deliberately tiny so CPU training takes ~1 min.
# ---------------------------------------------------------------------------
block_size = 64    # max context length T (positions 0..63)
n_embd = 64        # embedding width d_model
n_head = 4         # attention heads; head_dim = 64/4 = 16
n_layer = 2        # transformer blocks
batch_size = 32
max_steps = 400
lr = 3e-3


def get_batch():
    """Sample batch_size random crops of the corpus.
    x: (B, T) input chars;  y: (B, T) = x shifted left by one.
    Every position is a training example: predict char t+1 from chars <= t."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T)
    return x, y


# ---------------------------------------------------------------------------
# 3. Multi-head causal self-attention, by hand.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = n_embd // n_head
        # One fused linear computes Q, K, V for all heads at once:
        # (B,T,n_embd) -> (B,T,3*n_embd), then split.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)   # output mixing W_o
        # Causal mask: lower-triangular (T,T) of ones. register_buffer =
        # part of the module but not a parameter (no gradients).
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape                       # C = n_embd
        qkv = self.qkv(x)                       # (B, T, 3C)
        q, k, v = qkv.split(n_embd, dim=2)      # each (B, T, C)
        # Split channels into heads: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, self.head_dim).transpose(1, 2)

        # Attention scores: (B,h,T,d) @ (B,h,d,T) -> (B,h,T,T)
        # Scale by 1/sqrt(head_dim) so scores have ~unit variance at init
        # (Var(<q,k>) = head_dim for unit-variance entries).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask: position i may not see j > i. -inf -> softmax weight 0.
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)            # rows sum to 1: (B,h,T,T)
        y = att @ v                             # (B,h,T,T)@(B,h,T,d)->(B,h,T,d)
        # Merge heads back: (B,h,T,d) -> (B,T,h,d) -> (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)                     # (B, T, C)


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, nonlinearity, project back.
    This is where per-token 'computation' happens; attention only moves
    information BETWEEN tokens."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)   # (C) -> (4C)
        self.proj = nn.Linear(4 * n_embd, n_embd) # (4C) -> (C)

    def forward(self, x):                          # (B, T, C) -> (B, T, C)
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-LN transformer block: normalize, transform, add back (residual).
    The residual stream is the 'highway': gradients flow through the plain
    additions untouched, so depth doesn't kill optimization."""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):                          # (B, T, C)
        x = x + self.attn(self.ln1(x))             # communicate
        x = x + self.mlp(self.ln2(x))              # compute
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # (V, C) lookup table
        self.pos_emb = nn.Embedding(block_size, n_embd)   # (T_max, C) learned
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)          # (C) -> (V) logits

    def forward(self, idx, targets=None):
        # idx: (B, T) integer char ids
        B, T = idx.shape
        tok = self.tok_emb(idx)                            # (B, T, C)
        pos = self.pos_emb(torch.arange(T))                # (T, C)
        x = tok + pos                                      # broadcast -> (B,T,C)
        # Without pos_emb, attention is permutation-equivariant: it could
        # not tell "cat sat" from "sat cat".
        for block in self.blocks:
            x = block(x)                                   # (B, T, C)
        x = self.ln_f(x)                                   # (B, T, C)
        logits = self.head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            # Cross-entropy over all B*T positions at once. This is just
            # V-way classification: which character comes next?
            loss = F.cross_entropy(
                logits.view(B * T, vocab_size), targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx: (B, T0) prompt. Autoregressive loop: sample, append, repeat.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]      # crop to context window
            logits, _ = self(idx_cond)           # (B, T, V)
            logits = logits[:, -1, :] / temperature   # (B, V): last position
            probs = F.softmax(logits, dim=-1)    # (B, V)
            nxt = torch.multinomial(probs, 1)    # (B, 1) sample
            idx = torch.cat([idx, nxt], dim=1)   # (B, T+1)
        return idx


# ---------------------------------------------------------------------------
# 4. Train.
# ---------------------------------------------------------------------------
def sample_text(model, n=250):
    start = torch.zeros((1, 1), dtype=torch.long)   # start from token 0 ('\n')
    out = model.generate(start, n, temperature=0.8)
    return decode(out[0].tolist())


def main():
    model = MiniGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    # Expected loss at init: uniform over vocab -> -ln(1/V) = ln(V).
    print(f"expected initial loss ~ ln({vocab_size}) = "
          f"{math.log(vocab_size):.3f}\n")

    print("--- sample BEFORE training (should be uniform noise) ---")
    print(sample_text(model))
    print("--------------------------------------------------------\n")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    first_loss = None
    for step in range(max_steps):
        x, y = get_batch()                 # (B,T), (B,T)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % 50 == 0 or step == max_steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")
    dt = time.time() - t0
    print(f"\ntrained {max_steps} steps in {dt:.1f}s "
          f"({dt / max_steps * 1000:.0f} ms/step)")
    assert loss.item() < 0.6 * first_loss, "loss did not decrease enough"

    print("\n--- sample AFTER training (English-ish character statistics) ---")
    print(sample_text(model))
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    main()
