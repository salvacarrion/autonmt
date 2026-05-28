import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Reference: Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019
    (arXiv:1910.07467).

    Like ``nn.LayerNorm`` but drops the mean-centering step and the bias
    parameter, normalizing only by the root mean square::

        y = x / sqrt(mean(x**2) + eps) * weight

    Faster than LayerNorm in practice and empirically comparable or better.
    Used in T5 and most modern LLMs (LLaMA, Mistral, Qwen, ...).
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # rsqrt in fp32 then cast back: matches the LLaMA reference and avoids
        # underflow on fp16/bf16 activations.
        dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms).to(dtype) * self.weight
