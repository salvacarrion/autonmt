import torch
import torch.utils.data as tud
import tqdm


def greedy_search(model, dataset, sos_id, eos_id, batch_size, max_tokens, max_gen_length, **kwargs):
    model.eval()
    device = next(model.parameters()).device

    # Create dataloader
    collate_fn = lambda x: dataset.collate_fn(x, max_tokens=max_tokens)
    eval_dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    idxs = []
    probabilities = []
    with torch.no_grad():
        for x, _ in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
            # Move to device
            x = x.to(device)

            # Set start token <s> and initial probabilities
            # Sentence generated
            dec_idxs = torch.full((x.shape[0], 1), sos_id, dtype=torch.long).to(device)  # Sentence tokens
            dec_probs = torch.zeros(x.shape[0]).to(device)  # Sentence probability

            # Iterative decoder
            all_eos = False
            while not all_eos and dec_idxs.shape[1] <= max_gen_length:
                # Get next token (probs + idx)
                next_probabilities = model.forward(x, dec_idxs)[:, -1].log_softmax(-1)
                next_max_probabilities, next_max_idxs = next_probabilities.max(-1)

                # Concat new tokens with previous tokens
                next_max_idxs = next_max_idxs.unsqueeze(-1)
                dec_idxs = torch.cat((dec_idxs, next_max_idxs), axis=1)
                dec_probs += next_max_probabilities  # Sentence probability

                # Check if all sentences have an <eos>
                if bool((dec_idxs == eos_id).sum(axis=1).bool().all()):
                    break

            # Store batch results
            idxs.append(dec_idxs)
            probabilities.append(dec_probs)

    # Prettify output
    idxs = [item for batch_idxs in idxs for item in batch_idxs.tolist()]
    probabilities = torch.concat(probabilities)
    return idxs, probabilities


def beam_search(model, dataset, sos_id, eos_id, batch_size, max_tokens, max_gen_length, beam_width, **kwargs):
    model.eval()
    device = next(model.parameters()).device

    # Create dataloader
    def collate_fn(x):
        return dataset.collate_fn(x, max_tokens=max_tokens)
    eval_dataloader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    idxs = []
    probabilities = []
    vocab_size = len(dataset.trg_vocab)
    with torch.no_grad():
        for x, _ in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
            # Move to device
            x = x.to(device)

            # Set start token <s> and initial probabilities
            # Sentence generated
            dec_idxs = torch.full((x.shape[0], 1), sos_id, dtype=torch.long).to(device)  # Sentence tokens
            # dec_probs = torch.zeros(x.shape[0]).to(device)  # Sentence probability

            # Get top k word predictions
            next_probabilities = model.forward(x, dec_idxs)[:, -1, :]
            topk_next_probabilities, topk_next_idxs = next_probabilities.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)

            # Create top k hypothesis: Dec => [Dec + k0, Dec + k1, Dec + k3]
            dec_idxs = dec_idxs.repeat((beam_width, 1))  # [dec, dec, dec]
            topk_next_idxs = topk_next_idxs.reshape(-1, 1)
            dec_idxs = torch.cat((dec_idxs, topk_next_idxs), axis=-1)

            # Iterative decoder
            all_eos = False
            while not all_eos and dec_idxs.shape[1] <= max_gen_length:
                # Create dataset of hypotheses: X=(batch*beams, lengths) and Y = (batch*beams, lengths)
                beam_dataset = tud.TensorDataset(x.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=1), dec_idxs)
                tensor_dataloader = tud.DataLoader(beam_dataset, batch_size=batch_size)

                # Get next probs for each beam
                beam_next_probabilities = []
                for _x, _y in tensor_dataloader:
                    beam_next_probabilities.append(model.forward(_x, _y)[:, -1, :].log_softmax(-1))

                # Concat probabilities and reshape them => (batch, beams, probs)
                beam_next_probabilities = torch.cat(beam_next_probabilities, axis=0)
                beam_next_probabilities = beam_next_probabilities.reshape((-1, beam_width, beam_next_probabilities.shape[-1]))

                # Add top k probabilities
                topk_next_probabilities = topk_next_probabilities.unsqueeze(-1) + beam_next_probabilities
                topk_next_probabilities = topk_next_probabilities.flatten(start_dim=1)
                topk_next_probabilities, topk_idxs = topk_next_probabilities.topk(k=beam_width, axis=-1)

                next_chars = torch.remainder(topk_idxs, vocab_size).flatten().unsqueeze(-1)
                best_candidates = (topk_idxs / vocab_size).long()
                best_candidates += torch.arange(dec_idxs.shape[0] // beam_width, device=x.device).unsqueeze(-1) * beam_width
                dec_idxs = dec_idxs[best_candidates].flatten(end_dim=-2)
                dec_idxs = torch.cat((dec_idxs, next_chars), axis=1)

                # Check if all sentences have an <eos>
                if bool((next_chars == eos_id).sum(axis=1).bool().all()):
                    break

            # Format indices
            dec_idxs = dec_idxs.reshape(-1, beam_width, dec_idxs.shape[-1])

            # Store batch results
            idxs.append(dec_idxs[:, 0])
            probabilities.append(topk_next_probabilities[:, 0])

    # Prettify output
    idxs = [item for batch_idxs in idxs for item in batch_idxs.tolist()]
    probabilities = torch.concat(probabilities)
    return idxs, probabilities
