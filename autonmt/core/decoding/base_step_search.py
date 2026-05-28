from abc import abstractmethod

import torch
import torch.utils.data as tud
import tqdm

from autonmt.core.decoding.base_search import BaseSearch


class BaseStepSearch(BaseSearch):
    """Shared scaffolding for one-token-per-step decoders.

    Greedy and sampling-based strategies (temperature, top-k, top-p/nucleus)
    only differ in *how* the next token is picked from the decoder's logits at
    each step — the rest (DataLoader, encoder call, EOS short-circuit, length
    cap, output assembly) is identical. Subclasses implement
    :py:meth:`pick_next_token`; this class drives the loop.

    If the model exposes ``supports_incremental_decoding = True``, decoding
    switches to KV-cached mode: each step feeds only the last token plus an
    ``incremental_state`` dict, so the decoder cost per step is O(L) instead
    of O(L^2). Models without the flag get the legacy full-prefix path.
    """

    @abstractmethod
    def pick_next_token(self, logits):
        """Choose the next token per sequence.

        ``logits`` has shape ``(B, V)`` (unnormalized scores at the current
        position). Return a ``LongTensor`` of shape ``(B,)`` with the chosen
        token id per row.
        """
        ...

    def decode(self, model, dataset, sos_id, eos_id, pad_id, batch_size,
               max_tokens, max_len_a, max_len_b, num_workers, **kwargs):
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

        outputs = []
        with torch.no_grad():
            for (x, _), (x_len, _) in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
                max_gen_length = int(max_len_a * x.shape[1] + max_len_b)

                _, states = model.forward_encoder(x=x.to(device), x_len=x_len.to(device))

                y_pred = torch.full((x.shape[0], max_gen_length), pad_id, dtype=torch.long).to(device)
                y_pred[:, 0] = sos_id

                x_pad_mask = (x != pad_id) if model.packed_sequence else None
                eos_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(device)
                # Incremental mode keeps a per-batch cache that grows by one
                # token per step. ``None`` opts out and falls back to the
                # legacy full-prefix path.
                incremental_state = {} if incremental else None
                # Probe ``eos_mask.all()`` only every ``EARLY_STOP_EVERY``
                # steps. A per-step ``.all()`` forces a CUDA sync, blocking
                # the stream and preventing overlap with the next decoder
                # forward — for short forward passes this dominates.
                EARLY_STOP_EVERY = 8
                max_iter = 0
                for i in range(1, max_gen_length):
                    max_iter = i
                    # Incremental: feed only the previous token; the cache
                    # provides the K/V for all earlier positions.
                    y_in = y_pred[:, i-1:i] if incremental else y_pred[:, :i]
                    outputs_t, states = model.forward_decoder(
                        y=y_in, y_len=None, states=states, x_pad_mask=x_pad_mask,
                        incremental_state=incremental_state)
                    next_tok = self.pick_next_token(outputs_t[:, -1, :])
                    y_pred[:, i] = next_tok

                    eos_mask |= (next_tok == eos_id)
                    if i % EARLY_STOP_EVERY == 0 and bool(eos_mask.all()):
                        break

                # Include position ``max_iter`` — that's the token written at the final
                # step, whether EOS (decode strips it) or the final word at the cap.
                outputs.extend(y_pred[:, :max_iter + 1].tolist())

        return outputs, None
