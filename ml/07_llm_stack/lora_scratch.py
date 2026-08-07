"""
lora_scratch.py — LoRA implemented by hand (no PEFT), verified against full
fine-tuning.

Run:  uv run python ml/07_llm_stack/lora_scratch.py     (CPU, < 1 min)

The claim to verify (notes.md §4.1): fine-tuning updates are effectively
LOW-RANK — you can adapt a frozen network to a shifted task by learning only
a rank-r correction

    W' = W + (alpha / r) * B @ A,   B: (d_out, r) zero-init, A: (r, d_in)

with a tiny fraction of the trainable parameters, and match full fine-tuning.

Setup:
  base task    : classify points inside vs outside an annulus (ring)
  shifted task : the same annulus, rotated-and-stretched inputs (a fixed
                 linear warp) — the FEATURES the base net learned are still
                 the right features, they just need re-routing. This is
                 exactly the regime where LoRA works (and why it works on
                 pretrained LMs: pretraining built the features).

We compare, on the shifted task:
  1. frozen base model (no adaptation)      — should be poor
  2. full fine-tune of a copy               — the reference
  3. frozen base + rank-r LoRA on each Linear — should match (2) with ~2-4%
     of the trainable parameters
"""

import torch
import torch.nn as nn

torch.manual_seed(0)
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# data: annulus classification, optionally through a fixed linear warp
# ---------------------------------------------------------------------------
D_IN = 16  # embed the 2-D problem in 16-D so the Linears are non-trivial
# random (fixed) lift from 2-D to 16-D, shared by both tasks
LIFT = torch.randn(2, D_IN) / 2 ** 0.5
# the task shift: a fixed rotation+stretch applied to the 2-D inputs
theta = torch.tensor(0.9)
WARP = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                     [torch.sin(theta), torch.cos(theta)]]) @ \
       torch.tensor([[1.6, 0.0], [0.0, 0.6]])


def make_data(n, shifted, gen):
    xy = torch.randn(n, 2, generator=gen)
    r = xy.norm(dim=1)
    y = ((r > 0.7) & (r < 1.6)).float()        # inside the ring?
    if shifted:
        xy = xy @ WARP.T                        # warp the INPUTS only
    return xy @ LIFT, y


def accuracy(model, x, y):
    with torch.no_grad():
        return ((model(x).squeeze(-1) > 0).float() == y).float().mean().item()


# ---------------------------------------------------------------------------
# the base model: a small MLP
# ---------------------------------------------------------------------------
H = 256  # hidden width: wide enough that rank-4 << full rank, like a real net


def make_mlp():
    return nn.Sequential(
        nn.Linear(D_IN, H), nn.ReLU(),
        nn.Linear(H, H), nn.ReLU(),
        nn.Linear(H, 1),
    )


def train(model, params, x, y, steps=800, lr=1e-2):
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(x).squeeze(-1), y).backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# LoRA by hand: wrap a frozen nn.Linear with a rank-r bypass
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = x W^T + b  +  (alpha/r) * x A^T B^T   with W, b frozen.

    A: (r, d_in)  ~ N(0, 1/r)   (some nonzero init so A gets gradients)
    B: (d_out, r) = 0           => the bypass starts at EXACTLY zero, so the
                                   adapted model begins as the base model.
    (If both were zero, grad(A) ∝ B^T = 0 and nothing would ever train.)
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                      # freeze W, b
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) / r)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merge(self):
        """Fold BA into W — zero inference overhead (notes.md §4.1)."""
        with torch.no_grad():
            self.base.weight += self.scale * self.B @ self.A
        return self.base


def lorafy(model, r):
    """Replace every nn.Linear in a Sequential with a LoRA-wrapped copy."""
    return nn.Sequential(*[
        LoRALinear(m, r) if isinstance(m, nn.Linear) else m for m in model
    ])


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
def main():
    gen = torch.Generator().manual_seed(1)
    x_base, y_base = make_data(4000, shifted=False, gen=gen)
    x_shift, y_shift = make_data(4000, shifted=True, gen=gen)
    x_test, y_test = make_data(4000, shifted=True, gen=gen)

    # 1) pretrain the base model on the base task, then freeze it
    base = make_mlp()
    train(base, base.parameters(), x_base, y_base)
    base_acc_on_base = accuracy(base, x_base, y_base)
    base_acc_on_shift = accuracy(base, x_test, y_test)

    # 2) full fine-tune (a copy) on the shifted task — the reference
    full = make_mlp()
    full.load_state_dict(base.state_dict())
    n_full = sum(p.numel() for p in full.parameters())
    train(full, full.parameters(), x_shift, y_shift)
    full_acc = accuracy(full, x_test, y_test)

    # 3) LoRA: freeze the base, train only rank-r bypasses
    R = 4
    lora = lorafy(base, r=R)
    n_lora = n_trainable(lora)
    assert all(not p.requires_grad for p in base.parameters())
    train(lora, [p for p in lora.parameters() if p.requires_grad],
          x_shift, y_shift)
    lora_acc = accuracy(lora, x_test, y_test)

    # 4) merge the adapters into the frozen weights: same architecture as the
    #    base model, identical outputs to the adapted model
    merged = nn.Sequential(*[
        m.merge() if isinstance(m, LoRALinear) else m for m in lora
    ])
    merged_acc = accuracy(merged, x_test, y_test)

    print("=" * 64)
    print("LoRA from scratch — annulus task, shifted inputs")
    print("=" * 64)
    print(f"base model on base task:            {base_acc_on_base:6.1%}")
    print(f"base model on SHIFTED task (frozen):{base_acc_on_shift:6.1%}"
          "   <- adaptation is needed")
    print("-" * 64)
    print(f"{'method':<24}{'trainable params':>18}{'shifted-task acc':>18}")
    print(f"{'full fine-tune':<24}{n_full:>18,}{full_acc:>17.1%}")
    print(f"{'LoRA (r=' + str(R) + ')':<24}{n_lora:>18,}{lora_acc:>17.1%}")
    print(f"{'LoRA merged into W':<24}{0:>18,}{merged_acc:>17.1%}")
    print("-" * 64)
    print(f"trainable-parameter ratio: {n_lora:,}/{n_full:,} = "
          f"{n_lora / n_full:.1%}  ({n_full / n_lora:.0f}x fewer)")
    # LoRA params per layer, spelled out (the interview arithmetic):
    #   layer (d_in -> d_out): A is r*d_in, B is d_out*r  =>  r*(d_in + d_out)
    per_layer = [f"({m.base.in_features}->{m.base.out_features}): "
                 f"{m.A.numel() + m.B.numel()}"
                 for m in lora if isinstance(m, LoRALinear)]
    print("per-layer LoRA params [r*(d_in+d_out)]: " + ", ".join(per_layer))

    assert lora_acc > full_acc - 0.03, "LoRA failed to match full fine-tune"
    assert abs(merged_acc - lora_acc) < 1e-6, "merge changed the function!"
    print(f"\nChecks passed: LoRA matches full fine-tuning within 3 points "
          f"with {n_lora / n_full:.0%} of the params,\nand merging BA into W "
          "is exact.")


if __name__ == "__main__":
    main()
