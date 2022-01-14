import torch
import torch.utils.data as tud
import tqdm


def greedy_search(model, dataset, sos_id, eos_id, batch_size, max_tokens, max_len_a, max_len_b, num_workers, **kwargs):
    model.eval()
    device = next(model.parameters()).device

    # Create dataloader
    collate_fn = lambda x: dataset.collate_fn(x, max_tokens=max_tokens)
    eval_dataloader = tud.DataLoader(dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size,
                                     num_workers=num_workers)

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

            # Run encoder
            memory = model.forward_encoder(x)

            # Iterative decoder
            all_eos = False
            max_gen_length = int(max_len_a*x.shape[1] + max_len_b)
            while not all_eos and dec_idxs.shape[1] <= max_gen_length:
                # Get next token (probs + idx)
                next_probabilities = model.forward_decoder(dec_idxs, memory)[:, -1].log_softmax(-1)
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
