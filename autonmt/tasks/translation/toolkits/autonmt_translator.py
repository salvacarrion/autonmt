import torch

from autonmt.tasks.translation.bundle.dataset import TranslationDataset
from autonmt.tasks.translation.models import Seq2Seq
from autonmt.tasks.translation.base import BaseTranslator
from autonmt.tasks.translation.bundle.search_algorithms import greedy_search
from autonmt.utils import *
from autonmt.cmd import cmd_tokenizers

from typing import Type


class Translator(BaseTranslator):

    def __init__(self, model: Type[Seq2Seq], search_algorithm=greedy_search, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model
        self.search_algorithm = search_algorithm

        # Translation datasets (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

    def _get_model(self, dts: TranslationDataset, **kwargs):
        # Get vocab sizes
        src_vocab_size = len(dts.src_vocab)
        trg_vocab_size = len(dts.trg_vocab)

        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')

        # Create and train model
        model = self.model(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, **kwargs).to(device)
        return model

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path,
                    trg_vocab_path, **kwargs):
        # Get vocabs
        src_vocab_path = src_vocab_path.strip() + ".vocab"
        trg_vocab_path = trg_vocab_path.strip() + ".vocab"

        # Create datasets
        if not kwargs.get("external_data"):  # Training
            self.train_tds = TranslationDataset(file_prefix=train_path, src_lang=src_lang, trg_lang=trg_lang,
                                                src_vocab_path=src_vocab_path, trg_vocab_path=trg_vocab_path)
            self.val_tds = TranslationDataset(file_prefix=val_path, src_lang=src_lang, trg_lang=trg_lang,
                                              src_vocab_path=src_vocab_path, trg_vocab_path=trg_vocab_path)
        else:  # Evaluation
            self.test_tds = TranslationDataset(file_prefix=test_path, src_lang=src_lang, trg_lang=trg_lang,
                                               src_vocab_path=src_vocab_path, trg_vocab_path=trg_vocab_path)

    def _train(self, checkpoints_path, logs_path, **kwargs):
        # Create and train model
        model = self._get_model(dts=self.train_tds, **kwargs)
        model.fit(self.train_tds, self.val_tds, checkpoints_path=checkpoints_path, logs_path=logs_path, **kwargs)

    def _translate(self, output_path, checkpoint_path, beam_width, max_gen_length, batch_size, max_tokens, **kwargs):
        # Load model
        model = self._get_model(dts=self.test_tds, **kwargs)
        model_state_dict = torch.load(checkpoint_path)
        model.load_state_dict(model_state_dict)

        # Iterative decoding
        predictions, log_probabilities = self.search_algorithm(model=model, dataset=self.test_tds,
                                                               sos_id=self.test_tds.src_vocab.sos_id,
                                                               eos_id=self.test_tds.src_vocab.eos_id,
                                                               batch_size=batch_size, max_tokens=max_tokens,
                                                               beam_width=beam_width, max_gen_length=max_gen_length)
        # Decode output
        self._postprocess_output(predictions=predictions, output_path=output_path)

    def _postprocess_output(self, predictions, output_path):
        # Decode sentences
        hyp_tok = [self.test_tds.trg_vocab.decode(tokens, remove_special_tokens=True) for tokens in predictions]

        # Write file: hyp
        for lines, fname in [(hyp_tok, "hyp.tok")]:
            with open(os.path.join(output_path, fname), 'w') as f:
                f.writelines([' '.join(tokens) + '\n' for tokens in lines])

        # Write files: src, ref
        for lines, fname in [(self.test_tds.src_lines, "src.tok"), (self.test_tds.trg_lines, "ref.tok")]:
            with open(os.path.join(output_path, fname), 'w') as f:
                f.writelines([line + '\n' for line in lines])
