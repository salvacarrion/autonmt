import torch
import torch.utils.data as tud
import tqdm

from autonmt.core.search.base_search import BaseSearch


class BeamSearch(BaseSearch):
    """Batched beam search aligned with the same model API as :class:`GreedySearch`.

    This pass is intentionally a straight, functional implementation — no KV
    cache, no per-step state reordering, no length normalization. Those are the
    efficiency pass. The decoder is re-fed the full hypothesis prefix at every
    step, and the encoder states stay fixed (the source is expanded up front so
    beam branching does not require touching the heterogeneous state structures
    of Transformer/RNN/Conv).
    """

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
                offsets = torch.arange(B, device=device).unsqueeze(-1) * beam_width  # (B, 1)

                for _ in range(1, max_gen_length):
                    outputs, _ = model.forward_decoder(
                        y=dec_idxs, y_len=None, states=states, x_pad_mask=x_pad_mask)
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

                    dec_idxs = torch.cat(
                        [dec_idxs[gather_idx], next_tokens.view(-1, 1)], dim=-1)
                    finished = finished[gather_idx] | (next_tokens.view(-1) == eos_id)

                    if finished.all():
                        break

                # Best beam per sentence.
                dec_idxs = dec_idxs.view(B, beam_width, -1)
                beam_scores = beam_scores.view(B, beam_width)
                best = beam_scores.argmax(dim=-1)
                arange_B = torch.arange(B, device=device)
                best_seqs = dec_idxs[arange_B, best]       # (B, L)
                best_scores = beam_scores[arange_B, best]  # (B,)

                all_idxs.append(best_seqs)
                all_scores.append(best_scores)

        flat_idxs = [seq for batch in all_idxs for seq in batch.tolist()]
        probs = torch.cat(all_scores) if all_scores else torch.empty(0)
        return flat_idxs, probs
