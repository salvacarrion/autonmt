from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn

from autonmt.bundle.metrics import _sacrebleu  # TODO: I don't like this
from autonmt.preprocessing.processors import decode_lines


class LitSeq2Seq(pl.LightningModule):

    def __init__(self, src_vocab_size, trg_vocab_size, padding_idx,
                 criterion="cross_entropy", learning_rate=0.001, optimizer="adam", weight_decay=0, **kwargs):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        # Hyperparams
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        # Set criterion
        self.criterion_fn = self.configure_criterion(padding_idx)

        # Other
        self.save_hyperparameters()
        self.best_scores = defaultdict(float)
        self.best_scores = defaultdict(float)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown value for optimizer")
        return optimizer

    def configure_criterion(self, padding_idx):
        if self.criterion == "cross_entropy":
            criterion_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)
        else:
            raise ValueError("Unknown value for optimizer")
        return criterion_fn

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        loss, _ = self._step(batch, batch_idx, log_prefix=f"train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        eval_prefix = "val"
        if dataloader_idx is not None:
            extra = 'all' if self._filter_eval[dataloader_idx] is None else '+'.join(self._filter_eval[dataloader_idx])
            eval_prefix += "_" + extra
        loss, outputs = self._step(batch, batch_idx, log_prefix=eval_prefix)
        return loss, outputs

    def validation_epoch_end(self, outputs):
        # Print samples
        if self._print_samples:
            iter_yield = zip([outputs], ["all"]) if len(self._filter_eval) <= 1 else zip(outputs, self._filter_eval)
            for i, (preds, ts_filter) in enumerate(iter_yield):  # Iterate over dataloader
                print(f"=> Printing samples: (Filter: {'+'.join(ts_filter) if ts_filter else str(ts_filter)}; val. dataloader_idx={i})")
                src, hyp, ref = list(zip(*[(x["src"], x["hyp"], x["ref"]) for x in list(zip(*preds))[1]]))
                src, hyp, ref = sum(src, []), sum(hyp, []), sum(ref, [])
                for i, (src_i, hyp_i, ref_i) in enumerate(list(zip(src, hyp, ref))[:self._print_samples], 1):
                    print(f"- Src. #{i}: {src_i}")
                    print(f"- Ref. #{i}: {ref_i}")
                    print(f"- Hyp. #{i}: {hyp_i}")
                    print("")
                print("-"*100)

    def _step(self, batch, batch_idx, log_prefix):
        x, y = batch

        # Forward
        output = self.forward_encoder(x)
        output = self.forward_decoder(y, output)  # (B, L, E)

        # Compute loss
        output = output.transpose(1, 2)[:, :, :-1]  # Remove last index to match shape with 'y[1:]'
        y = y[:, 1:]  # Remove <sos>
        loss = self.criterion_fn(output, y)

        # Metrics: Accuracy
        predictions = output.detach().argmax(1)
        batch_errors = (predictions != y).sum().item()
        accuracy = 1 - (batch_errors / predictions.numel())

        # Log params
        self.log(f"{log_prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{log_prefix}_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Compute metrics for validaiton
        outputs = None
        if log_prefix.startswith("val"):
            outputs = self._compute_metrics(y_hat=predictions, y=y, metrics={"bleu"}, x=x, log_prefix=log_prefix)
        return loss, outputs

    def _compute_metrics(self, y_hat, y, x, metrics, log_prefix):
        # Decode lines
        # Since ref lines are encoded, unknowns can appear. Therefore, for small vocabularies the scores could be strongly biased
        hyp_lines = [self._trg_vocab.decode(list(x)) for x in y_hat.detach().cpu().numpy()]
        ref_lines = [self._trg_vocab.decode(list(x)) for x in y.detach().cpu().numpy()]
        src_lines = [self._src_vocab.decode(list(x)) for x in x.detach().cpu().numpy()]

        # Full decoding
        hyp_lines = decode_lines(hyp_lines, self._trg_vocab.lang, self._subword_model, self._pretok_flag, self._trg_model_vocab_path,  remove_unk_hyphen=True)
        ref_lines = decode_lines(ref_lines, self._trg_vocab.lang, self._subword_model, self._pretok_flag, self._trg_model_vocab_path,  remove_unk_hyphen=True)
        src_lines = decode_lines(src_lines, self._src_vocab.lang, self._subword_model, self._pretok_flag, self._src_model_vocab_path,  remove_unk_hyphen=True)

        # Compute metrics
        scores = []

        # Compute sacrebleu
        scores += _sacrebleu(hyp_lines=hyp_lines, ref_lines=ref_lines, metrics=metrics)

        # Log metrics
        for score in scores:
            # Get score and keep best score
            score_name = score['name'].lower()
            best_score = max(score['score'], self.best_scores[score_name])
            self.best_scores[score_name] = best_score

            self.log(f"{log_prefix}_{score_name}", score['score'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{log_prefix}_best_{score_name}", best_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"hyp": hyp_lines, "ref": ref_lines, "src": src_lines}
