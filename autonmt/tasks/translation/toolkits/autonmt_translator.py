import torch

from autonmt.tasks.translation.bundle.dataset import TranslationDataset
from autonmt.tasks.translation.models import Seq2Seq
from autonmt.tasks.translation.base import BaseTranslator
from autonmt.utils import *

from typing import Type


class Translator(BaseTranslator):

    def __init__(self, model: Type[Seq2Seq], **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model

        # Translation datasets (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

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
        # Create paths
        make_dir([checkpoints_path, logs_path])

        # Get vocab sizes
        src_vocab_size = len(self.train_tds.src_vocab)
        trg_vocab_size = len(self.train_tds.trg_vocab)

        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')

        # Create and train model
        model = self.model(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, **kwargs).to(device)
        model.fit(self.train_tds, self.val_tds, checkpoints_path=checkpoints_path, logs_path=logs_path, **kwargs)

    def _translate(self, src_lang, trg_lang, data_path, output_path, checkpoint_path, src_spm_model_path,
                   trg_spm_model_path, beam_width, max_gen_length,
                   *args, **kwargs):
        qwe = 33
        pass
