import torch

from autonmt.tasks.translation.bundle.translation_dataset import TranslationDataset
from autonmt.tasks.translation.bundle.vocabulary import VocabularyBytes
from autonmt.tasks.translation.models import Seq2Seq
from autonmt.tasks.translation.base import BaseTranslator
from autonmt.tasks.translation.bundle.search_algorithms import greedy_search, beam_search
from autonmt.utils import *

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
        padding_idx = dts.src_vocab.pad_id
        assert dts.src_vocab.pad_id == dts.trg_vocab.pad_id
        model = self.model(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size, padding_idx=padding_idx,
                           **kwargs).to(device)

        # Count parameters
        trainable_params, non_trainable_params = self._count_model_parameters(model)
        print(f"\t - [INFO]: Total trainable parameters: {trainable_params:,}")
        print(f"\t - [INFO]: Total non-trainable parameters: {non_trainable_params:,}")
        return model

    def _count_model_parameters(self, model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path,
                    trg_vocab_path, subword_model, **kwargs):

        # Get vocabs
        if subword_model in {"bytes"}:
            src_vocab = VocabularyBytes()
            trg_vocab = VocabularyBytes()
        else:
            # Do not use *.vocabf if possible. It could be not equivalent to *.vocab
            src_vocab = src_vocab_path + ".vocab" if src_vocab_path else None
            trg_vocab = trg_vocab_path + ".vocab" if trg_vocab_path else None

        # Create datasets
        if not kwargs.get("external_data"):  # Training
            self.train_tds = TranslationDataset(file_prefix=train_path, src_lang=src_lang, trg_lang=trg_lang,
                                                src_vocab=src_vocab, trg_vocab=trg_vocab)
            self.val_tds = TranslationDataset(file_prefix=val_path, src_lang=src_lang, trg_lang=trg_lang,
                                              src_vocab=self.train_tds.src_vocab, trg_vocab=self.train_tds.trg_vocab)
        else:  # Evaluation
            # Check vocab values
            if src_vocab is None or trg_vocab is None:
                raise ValueError("'src_vocab_path' and 'trg_vocab_path' cannot be 'None' during testing")

            self.test_tds = TranslationDataset(file_prefix=test_path, src_lang=src_lang, trg_lang=trg_lang,
                                               src_vocab=src_vocab, trg_vocab=trg_vocab)

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
        # Decode: hyp
        hyp_tok = [self.test_tds.trg_vocab.decode(tokens) for tokens in predictions]

        # Decode: src
        src_tok = []
        for line in self.test_tds.src_lines:
            line_enc = self.test_tds.src_vocab.encode(line)
            line_dec = self.test_tds.src_vocab.decode(line_enc)
            src_tok.append(line_dec)

        # Decode: ref
        ref_tok = []
        for line in self.test_tds.trg_lines:
            line_enc = self.test_tds.trg_vocab.encode(line)
            line_dec = self.test_tds.trg_vocab.decode(line_enc)
            ref_tok.append(line_dec)

        # Write file: hyp, src, ref
        for lines, fname in [(hyp_tok, "hyp.tok"), (src_tok, "src.tok"), (ref_tok, "ref.tok")]:
            write_file_lines(lines=lines, filename=os.path.join(output_path, fname))
