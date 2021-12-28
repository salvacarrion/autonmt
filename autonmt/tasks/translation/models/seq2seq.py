import os

import torch
import torch.nn as nn
import torch.utils.data as tud
import tqdm
from autonmt.tasks.translation.bundle.translation_dataset import TranslationDataset
from abc import ABC, abstractmethod


class Seq2Seq(nn.Module, ABC):

    def __init__(self, src_vocab_size, trg_vocab_size, **kwargs):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def fit(self, ds_train: TranslationDataset, ds_val: TranslationDataset,
            batch_size, max_tokens, max_epochs, learning_rate, weight_decay, clip_norm,
            patience, checkpoints_path, logs_path, **kwargs):

        # Checks: checkpoints_path
        if not checkpoints_path:
            print("\t- [WARNING]: No 'checkpoints_path' was specified")

        # Get device
        device = next(self.parameters()).device

        # Create dataloaders
        collate_fn = lambda x: ds_train.collate_fn(x, max_tokens=max_tokens)
        train_dataloader = tud.DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Criterion and Optimizer
        assert ds_train.src_vocab.pad_id == ds_train.src_vocab.pad_id
        pad_idx = ds_train.src_vocab.pad_id
        criterion_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train loop
        best_loss = float(torch.inf)
        for i in range(max_epochs):
            self.train()
            print(f"Epoch #{i+1}:")
            train_losses, train_errors, train_sizes = [], [], []
            for x, y in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
                # Move to device
                x, y = x.to(device), y.to(device)

                # Forward
                probs = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]

                # Backward
                loss = criterion_fn(probs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_norm)  # Clip norm
                optimizer.step()
                optimizer.zero_grad()

                # compute accuracy
                predictions = probs.argmax(1)
                batch_errors = (predictions != y)

                # To keep track
                train_errors.append(batch_errors.sum().item())
                train_sizes.append(batch_errors.numel())

                # Keep results
                train_losses.append(loss.item())

            # Compute metrics
            train_loss = sum(train_losses) / len(train_losses)
            train_acc = 1 - sum(train_errors) / sum(train_sizes)
            print("\t- train_loss={:.3f}; train_acc={:.3f}".format(train_loss, train_acc))

            # Validation
            val_loss, val_acc = self.evaluate(ds_val, batch_size=batch_size, max_tokens=max_tokens,
                                              device=device, prefix="val")

            # Save model
            if checkpoints_path is not None:
                print("\t- Saving checkpoint...")
                torch.save(self.state_dict(), os.path.join(checkpoints_path, "checkpoint_last.pt"))

                # Save best
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.state_dict(), os.path.join(checkpoints_path, "checkpoint_best.pt"))

            # # Validation with beam search
            # predictions, log_probabilities = seq2seq.beam_search(model, X_new)
            # output = [target_index.tensor2text(p) for p in predictions]

    def evaluate(self, eval_ds, batch_size=128, max_tokens=None, prefix="eval", **kwargs):
        self.eval()
        device = next(self.parameters()).device  # Get device

        # Create dataloader
        collate_fn = lambda x: eval_ds.collate_fn(x, max_tokens=max_tokens)
        eval_dataloader = tud.DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Criterion and Optimizer
        assert eval_ds.src_vocab.pad_id == eval_ds.src_vocab.pad_id
        pad_idx = eval_ds.src_vocab.pad_id
        criterion_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

        with torch.no_grad():
            eval_losses, eval_errors, eval_sizes = [], [], []
            for x, y in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
                # Move to device
                x, y = x.to(device), y.to(device)

                # Forward
                probs = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]

                # Get loss
                loss = criterion_fn(probs, y)

                # compute accuracy
                predictions = probs.argmax(1)
                batch_errors = (predictions != y)

                # To keep track
                eval_errors.append(batch_errors.sum().item())
                eval_sizes.append(batch_errors.numel())

                # Keep results
                eval_losses.append(loss.item())

            # Compute metrics
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_acc = 1 - sum(eval_errors) / sum(eval_sizes)
            print("\t- {}_loss={:.3f}; {}_acc={:.3f}".format(prefix, eval_loss, prefix, eval_acc))
        return eval_loss, eval_acc

