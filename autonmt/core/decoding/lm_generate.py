"""Prompt-conditioned autoregressive generation for decoder-only LMs.

The seq2seq search loop in :mod:`autonmt.core.decoding.base_step_search` is
encoder-seeded (it calls ``forward_encoder`` and starts from a single ``<sos>``),
so it doesn't fit a decoder-only LM. This is the LM analogue: it prefills the
prompt into the model's KV cache, then samples one token at a time — **reusing
the exact same ``pick_next_token`` strategies** (greedy / top-k / top-p /
multinomial) so there is no second copy of the sampling logic.
"""
import torch

from autonmt.core.decoding import GreedySearch


@torch.no_grad()
def lm_generate(model, prompt_ids, sampler=None, max_new_tokens=64,
                eos_id=None, device=None):
    """Generate a continuation for a single prompt.

    Parameters
    ----------
    model : LitLM
        A decoder-only model whose ``forward(x, incremental_state=...)`` returns
        logits ``(B, L, V)`` and supports KV-cached decoding.
    prompt_ids : list of int
        The seed token ids (e.g. ``corpus.encode(prompt, add_sos=True)``).
    sampler : BaseStepSearch, optional
        Any object with ``pick_next_token(logits) -> (B,)``. Defaults to
        :class:`GreedySearch`. Pass ``TopKSampling`` / ``TopPSampling`` /
        ``MultinomialSampling`` for stochastic decoding.
    max_new_tokens : int, default 64
        Maximum number of tokens to generate.
    eos_id : int, optional
        Stop early when this token is produced (it is included in the output).
    device : torch.device, optional
        Defaults to the model's device.

    Returns
    -------
    list of int
        The generated token ids (not including the prompt).
    """
    model.eval()
    sampler = sampler if sampler is not None else GreedySearch()
    device = device if device is not None else next(model.parameters()).device
    max_seq_len = getattr(model, "max_seq_len", None)

    incremental_state = {}
    x = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)  # (1, L)
    logits = model(x, incremental_state=incremental_state)
    next_logits = logits[:, -1, :]                                         # (1, V)

    generated = []
    cur_len = x.shape[1]
    for _ in range(max_new_tokens):
        token = sampler.pick_next_token(next_logits)        # (1,)
        token_id = int(token.item())
        generated.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break
        cur_len += 1
        if max_seq_len is not None and cur_len >= max_seq_len:
            break
        step_logits = model(token.view(1, 1), incremental_state=incremental_state)
        next_logits = step_logits[:, -1, :]

    return generated
