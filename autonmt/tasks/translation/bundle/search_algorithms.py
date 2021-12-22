import torch
import torch.utils.data as tud
import tqdm


def greedy_search(model, ds, sos_id, eos_id, batch_size, max_tokens, max_length, **kwargs):
    model.eval()
    device = next(model.parameters()).device

    # Create dataloader
    collate_fn = lambda x: ds.collate_fn(x, max_tokens=max_tokens)
    eval_dataloader = tud.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
            while not all_eos and dec_idxs.shape[1] <= max_length:
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

