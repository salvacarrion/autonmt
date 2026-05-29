import torch
import torch.utils.data as tud
import tqdm

from autonmt.core.decoding.base_search import BaseSearch


def _reorder_states(states, new_order):
    """Recursively walk a (possibly nested) state structure and ``index_select``
    each tensor by ``new_order`` along the first dim whose size matches
    ``new_order``'s length. Non-tensor leaves pass through unchanged.

    Best-effort heuristic — relies on each state tensor having a batch-like
    dimension equal to ``len(new_order)``. For the models in this repo
    (Transformer encoder state with batch dim 1, RNN hidden with batch dim 1,
    attention-RNN ``enc_outputs`` with batch dim 0) it picks the right one. If
    two dims happen to match, the first wins.

    TODO: this is fragile. Properly fixing it requires each model to declare
    which dim is the batch axis for every tensor in its state (e.g. via a
    ``reorder_state`` method on ``LitSeq2Seq`` or a per-tensor annotation).
    Until then, edge cases where another dim happens to equal B*K (e.g. small
    batches with vocab/length collisions) will silently reorder the wrong axis.
    """
    if states is None:
        return states
    if isinstance(states, torch.Tensor):
        bk = new_order.shape[0]
        for dim in range(states.dim()):
            if states.size(dim) == bk:
                return states.index_select(dim, new_order)
        return states
    if isinstance(states, (tuple, list)):
        cls = type(states)
        return cls(_reorder_states(s, new_order) for s in states)
    if isinstance(states, dict):
        # ``incremental_state`` dicts can contain non-tensor scalars (e.g. a
        # 'step' counter) — those pass through via the leaf branches above.
        return {k: _reorder_states(v, new_order) for k, v in states.items()}
    return states


# Keys in the KV cache whose value is constant across the K beams of a sentence
# (cross-attention K/V derived from the encoder memory, which itself is
# replicated K times via repeat_interleave). Reordering them by ``gather_idx``
# permutes identical rows — a semantic no-op that wastes GPU memory bandwidth.
_INVARIANT_CACHE_KEYS = frozenset({"cross_k", "cross_v"})


def _reorder_incremental_state(incremental_state, new_order):
    """In-place reorder of the per-layer KV cache by parent-beam index.

    Skips :data:`_INVARIANT_CACHE_KEYS` — those tensors are constant across
    the K beams of each sentence, so permuting them is wasted work.
    """
    if incremental_state is None:
        return
    for v in incremental_state.values():
        if not isinstance(v, dict):
            continue
        for sub_k, sub_v in v.items():
            if sub_k in _INVARIANT_CACHE_KEYS or not isinstance(sub_v, torch.Tensor):
                continue
            bk = new_order.shape[0]
            for dim in range(sub_v.dim()):
                if sub_v.size(dim) == bk:
                    v[sub_k] = sub_v.index_select(dim, new_order)
                    break


class BeamSearch(BaseSearch):
    """Batched beam search aligned with the same model API as :class:`GreedySearch`.

    Per-step state propagation: the ``states`` returned by ``forward_decoder``
    are reordered by ``gather_idx`` each step. For stateless decoders (Transformer)
    this is a no-op since encoder states are constant; for stateful decoders (RNN)
    it is what makes beam>1 correct.

    Final beam selection uses length-normalized scores: ``score / L^length_penalty``.
    Set ``length_penalty=0`` for raw log-probabilities (favors short hypotheses).
    """

    def __init__(self, length_penalty=1.0):
        assert length_penalty >= 0
        self.length_penalty = length_penalty

    def decode(self, model, dataset, sos_id, eos_id, pad_id, batch_size,
               max_tokens, max_len_a, max_len_b, num_workers, *,
               beam_width, **kwargs):
        # Algorithm at a glance:
        #   B  = batch size (sentences per mini-batch)
        #   K  = beam_width (live hypotheses per sentence)
        #   BK = B * K     (rows in every per-beam tensor)
        #   V  = target vocab size
        # Layout used everywhere: beam i of sentence j sits at row j*K + i
        # (interleaved). ``offsets[j] = j*K`` lets us map (sentence, local-beam)
        # → absolute row.
        assert beam_width >= 1
        model.eval()
        device = next(model.parameters()).device
        pin_memory = device.type == "cuda"
        incremental = getattr(model, "supports_incremental_decoding", False)

        eval_dataloader = tud.DataLoader(
            dataset,
            collate_fn=dataset.get_collate_fn(max_tokens),
            num_workers=num_workers, persistent_workers=bool(num_workers),
            pin_memory=pin_memory,
            batch_size=batch_size, shuffle=False,
        )

        V = len(dataset.tgt_vocab)
        K = beam_width
        all_idxs = []
        all_scores = []

        with torch.no_grad():
            for (x, _), (x_len, _) in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
                x = x.to(device)                              # (B, L_src)
                x_len = x_len.to(device)                      # (B,)
                B = x.shape[0]
                BK = B * K
                max_gen_length = int(max_len_a * x.shape[1] + max_len_b)

                # Expand the source so the encoder produces BK identical copies
                # of each sentence's memory. Done up front so the algorithm can
                # stay model-agnostic — no need to broadcast/reshape the
                # heterogeneous state structures of different decoder families.
                x_exp = x.repeat_interleave(K, dim=0)         # (BK, L_src)
                x_len_exp = x_len.repeat_interleave(K, dim=0) # (BK,)
                _, states = model.forward_encoder(x=x_exp, x_len=x_len_exp)
                x_pad_mask = (x_exp != pad_id) if model.packed_sequence else None

                # Every beam starts at <sos>.
                dec_idxs = torch.full((BK, 1), sos_id, dtype=torch.long, device=device)

                # Initial scores: only beam 0 of each sentence is "real" at
                # step 1; the others are dead (-inf) so the first top-K picks
                # K *distinct* extensions of <sos> instead of K copies of the
                # single best one.
                beam_scores = torch.full((B, K), float("-inf"), device=device)
                beam_scores[:, 0] = 0.0
                beam_scores = beam_scores.view(-1)             # (BK,)

                finished = torch.zeros(BK, dtype=torch.bool, device=device)
                beam_lengths = torch.zeros(BK, dtype=torch.long, device=device)
                offsets = torch.arange(B, device=device).unsqueeze(-1) * K  # (B, 1)
                # Incremental mode: each layer holds K/V caches that grow by
                # one token per step. None opts out → legacy full-prefix path.
                incremental_state = {} if incremental else None

                # Pre-build the "frozen" log-prob row: 0 at <pad>, -inf elsewhere.
                # Applied unconditionally via ``torch.where(finished, ...)`` so
                # we don't need ``if finished.any()`` (which forces a CUDA sync).
                frozen_logits = torch.full((V,), float("-inf"), device=device)
                frozen_logits[pad_id] = 0.0

                # Probe the early-stop condition only every ``EARLY_STOP_EVERY``
                # steps. Each ``.all()`` would otherwise bring a bool to host,
                # blocking the stream and preventing overlap with the next
                # decoder forward.
                EARLY_STOP_EVERY = 8

                for step in range(1, max_gen_length):
                    # Next-token log-probabilities for every live hypothesis.
                    # Incremental: feed only the last token; cache provides the rest.
                    y_in = dec_idxs[:, -1:] if incremental else dec_idxs
                    outputs, states = model.forward_decoder(
                        y=y_in, y_len=None, states=states, x_pad_mask=x_pad_mask,
                        incremental_state=incremental_state)
                    log_probs = outputs[:, -1, :].log_softmax(-1)   # (BK, V)

                    # Freeze finished hypotheses: force <pad> with log_prob 0 so
                    # their cumulative score is preserved and they cannot spawn
                    # new branches that would push live hypotheses out of top-K.
                    log_probs = torch.where(finished.unsqueeze(-1),
                                            frozen_logits.unsqueeze(0), log_probs)

                    # Add log-prob to running score, then flatten the (beam,
                    # vocab) axis to a single (K*V,) per sentence so top-K
                    # competes across both — i.e. the K best (parent, token)
                    # combinations across the K beams of each sentence.
                    scores = (beam_scores.unsqueeze(-1) + log_probs)  # (BK, V)
                    scores = scores.view(B, K * V)                    # (B, K*V)
                    beam_scores, top_idxs = scores.topk(K, dim=-1)    # both (B, K)
                    beam_scores = beam_scores.view(-1)                # (BK,)

                    # Each flat index in top_idxs packs (parent beam, token):
                    #   parent = idx // V, token = idx % V.
                    beam_source = top_idxs // V                       # (B, K)
                    next_tokens = top_idxs % V                        # (B, K)

                    # Convert per-sentence local beam index → absolute row in
                    # the (BK,) layout, so we can gather parent prefixes/states.
                    gather_idx = (beam_source + offsets).view(-1)     # (BK,)

                    # Bookkeeping: bump length only for hypotheses that were
                    # alive *before* this step (frozen <pad> beams keep their
                    # original length). Must run before updating ``finished``.
                    finished_before = finished[gather_idx]
                    beam_lengths = beam_lengths[gather_idx] + (~finished_before).long()

                    # Stitch each new token onto its parent's prefix.
                    dec_idxs = torch.cat(
                        [dec_idxs[gather_idx], next_tokens.view(-1, 1)], dim=-1)  # (BK, t+1)
                    finished = finished_before | (next_tokens.view(-1) == eos_id)

                    # Permute decoder states so row k carries the state of the
                    # *parent* hypothesis whose history flows into beam k next
                    # step. No-op for Transformer encoder memory (constant
                    # across the K beams of a sentence); critical for stateful
                    # decoders like RNN whose hidden state evolves per token —
                    # and for the KV cache when running incrementally.
                    states = _reorder_states(states, gather_idx)
                    # Reorder only the parts of the cache that actually depend
                    # on beam history (self_k/self_v); cross_k/cross_v are
                    # invariant across the K beams of each sentence — skipping
                    # them saves O(n_layers · L_src · BK · D) memory traffic
                    # per step.
                    _reorder_incremental_state(incremental_state, gather_idx)

                    if step % EARLY_STOP_EVERY == 0 and bool(finished.all()):
                        break

                # Pick the best beam per sentence using length-normalized score
                # (raw log-prob biases toward short hypotheses; Wu et al. 2016).
                lengths = beam_lengths.clamp(min=1).float().view(B, K)
                scores_2d = beam_scores.view(B, K)
                normalized = scores_2d / lengths.pow(self.length_penalty)

                dec_idxs = dec_idxs.view(B, K, -1)
                best = normalized.argmax(dim=-1)
                arange_B = torch.arange(B, device=device)
                best_seqs = dec_idxs[arange_B, best]       # (B, L_out)
                best_scores = scores_2d[arange_B, best]    # (B,) raw cumulative log-prob

                all_idxs.append(best_seqs)
                all_scores.append(best_scores)

        flat_idxs = [seq for batch in all_idxs for seq in batch.tolist()]
        probs = torch.cat(all_scores) if all_scores else torch.empty(0)
        return flat_idxs, probs
