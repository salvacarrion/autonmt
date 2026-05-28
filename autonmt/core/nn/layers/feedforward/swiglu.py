import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Reference: Shazeer, *GLU Variants Improve Transformer*, 2020
    (arXiv:2002.05202).

    Replaces the standard transformer FFN (Linear-ReLU-Linear) with a gated
    variant::

        FFN(x) = (Swish(x W_gate) * x W_up) W_down

    Three projections instead of two, but the gating empirically improves
    quality at parity. Used in PaLM, LLaMA, Mistral, Qwen, ...

    To keep parameter count comparable to a matching ReLU FFN, ``hidden_dim``
    is typically reduced by ~2/3 (e.g. ``int(2/3 * 4 * dim)`` rounded to a
    convenient multiple). Caller decides — this module just takes the value.
    """

    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
