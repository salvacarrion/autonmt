import torch
import torch.utils.data as tud
import tqdm


def greedy_search(model, dataset, sos_id, eos_id, pad_id, batch_size, max_tokens, max_len_a, max_len_b, num_workers, **kwargs):
    model.eval()
    device = next(model.parameters()).device
    pin_memory = False if device.type == "cpu" else True

    # Create dataloader
    eval_dataloader = tud.DataLoader(dataset,
                                     collate_fn=dataset.get_collate_fn(max_tokens),
                                     num_workers=num_workers, persistent_workers=bool(num_workers),
                                     pin_memory=pin_memory,
                                     batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        outputs = []
        for (x, _), (x_len, _) in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
            max_gen_length = int(max_len_a*x.shape[1] + max_len_b)

            # Run encoder
            _, states = model.forward_encoder(x=x.to(device), x_len=x_len.to(device))

            # Set start token <sos> and initial probabilities
            y_pred = torch.full((x.shape[0], max_gen_length), pad_id, dtype=torch.long).to(device)  # (B, L)
            y_pred[:, 0] = sos_id

            # Iterate over trg tokens
            x_pad_mask = (x != pad_id) if model.packed_sequence else None  # Mask padding
            eos_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(device)
            max_iter = 0
            for i in range(1, max_gen_length):
                max_iter = i
                outputs_t, states = model.forward_decoder(y=y_pred[:, :i], state=states, x_pad_mask=x_pad_mask)
                top1 = outputs_t[:, -1, :].argmax(1)  # Get most probable next-word (logits)

                # Update y_pred for next iteration
                y_pred[:, i] = top1

                # Check for EOS tokens
                eos_mask |= (top1 == eos_id)  # in-place OR

                # Break if all sentences have an EOS token
                if eos_mask.all():
                    break

            # Add outputs
            outputs.extend(y_pred[:, :max_iter].tolist())

    return outputs, None
