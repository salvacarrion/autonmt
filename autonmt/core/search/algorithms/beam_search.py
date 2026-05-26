import torch
import torch.utils.data as tud
import tqdm

from autonmt.core.search.base_search import BaseSearch


def _reorder_states(states, new_order):
    """Recursively walk a (possibly nested) state structure and ``index_select``
    each tensor by ``new_order`` along the first dim whose size matches
    ``new_order``'s length. Non-tensor leaves pass through unchanged.

    Best-effort heuristic — relies on each state tensor having a batch-like
    dimension equal to ``len(new_order)``. For the models in this repo
    (Transformer encoder state with batch dim 1, RNN hidden with batch dim 1,
    AttentionRNN ``enc_outputs`` with batch dim 0) it picks the right one. If
    two dims happen to match, the first wins.
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
    return states


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
        assert beam_width >= 1
        model.eval()
        device = next(model.parameters()).device
        pin_memory = device.type == "cuda"

        eval_dataloader = tud.DataLoader(
            dataset,
            collate_fn=dataset.get_collate_fn(max_tokens),
            num_workers=num_workers, persistent_workers=bool(num_workers),
            pin_memory=pin_memory,
            batch_size=batch_size, shuffle=False,
        )

        vocab_size = len(dataset.trg_vocab)
        all_idxs = []
        all_scores = []

        with torch.no_grad():
            for (x, _), (x_len, _) in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
                x = x.to(device)
                x_len = x_len.to(device)
                B = x.shape[0]
                BK = B * beam_width
                max_gen_length = int(max_len_a * x.shape[1] + max_len_b)

                # Expand source up front so every model's encoder state is implicitly
                # broadcast across beams. Layout used everywhere below: beam i of
                # sentence j lives at index j * beam_width + i (interleaved).
                x_exp = x.repeat_interleave(beam_width, dim=0)
                x_len_exp = x_len.repeat_interleave(beam_width, dim=0)
                _, states = model.forward_encoder(x=x_exp, x_len=x_len_exp)
                x_pad_mask = (x_exp != pad_id) if model.packed_sequence else None

                # Every beam starts at <sos>.
                dec_idxs = torch.full((BK, 1), sos_id, dtype=torch.long, device=device)

                # Mask all but beam 0 with -inf at step 1 so the first top-k picks
                # k *distinct* extensions of <sos> rather than k copies of the same.
                beam_scores = torch.full((B, beam_width), float("-inf"), device=device)
                beam_scores[:, 0] = 0.0
                beam_scores = beam_scores.view(-1)  # (BK,)

                finished = torch.zeros(BK, dtype=torch.bool, device=device)
                beam_lengths = torch.zeros(BK, dtype=torch.long, device=device)
                offsets = torch.arange(B, device=device).unsqueeze(-1) * beam_width  # (B, 1)

                for _ in range(1, max_gen_length):
                    outputs, states = model.forward_decoder(y=dec_idxs, y_len=None, states=states, x_pad_mask=x_pad_mask)
                    log_probs = outputs[:, -1, :].log_softmax(-1)  # (BK, V)

                    # Lock finished hypotheses: force <pad> with log_prob 0 so the
                    # cumulative score is preserved and no new branches are spawned.
                    if finished.any():
                        log_probs[finished] = float("-inf")
                        log_probs[finished, pad_id] = 0.0

                    scores = beam_scores.unsqueeze(-1) + log_probs               # (BK, V)
                    scores = scores.view(B, beam_width * vocab_size)             # (B, K*V)

                    beam_scores, top_idxs = scores.topk(beam_width, dim=-1)      # (B, K)
                    beam_scores = beam_scores.view(-1)                           # (BK,)

                    beam_source = top_idxs // vocab_size                         # (B, K)
                    next_tokens = top_idxs % vocab_size                          # (B, K)

                    # Absolute index in the (BK,) layout for the gather below.
                    gather_idx = (beam_source + offsets).view(-1)                # (BK,)

                    # Length bookkeeping: increment only for hypotheses that were
                    # still active before this step. Must be done *before* the
                    # ``finished`` update so already-finished beams don't tick up.
                    finished_before = finished[gather_idx]
                    beam_lengths = beam_lengths[gather_idx] + (~finished_before).long()

                    dec_idxs = torch.cat([dec_idxs[gather_idx], next_tokens.view(-1, 1)], dim=-1)
                    finished = finished_before | (next_tokens.view(-1) == eos_id)

                    # Reorder decoder states so beam k's history lives at row k
                    # in the next step (necessary for stateful decoders like RNN;
                    # no-op for Transformer where states are the encoder memory).
                    states = _reorder_states(states, gather_idx)

                    if finished.all():
                        break

                # Pick the best beam per sentence using length-normalized scores.
                lengths = beam_lengths.clamp(min=1).float().view(B, beam_width)
                scores_2d = beam_scores.view(B, beam_width)
                normalized = scores_2d / lengths.pow(self.length_penalty)

                dec_idxs = dec_idxs.view(B, beam_width, -1)
                best = normalized.argmax(dim=-1)
                arange_B = torch.arange(B, device=device)
                best_seqs = dec_idxs[arange_B, best]       # (B, L)
                best_scores = scores_2d[arange_B, best]    # (B,)  raw cumulative log-prob

                all_idxs.append(best_seqs)
                all_scores.append(best_scores)

        flat_idxs = [seq for batch in all_idxs for seq in batch.tolist()]
        probs = torch.cat(all_scores) if all_scores else torch.empty(0)
        return flat_idxs, probs
