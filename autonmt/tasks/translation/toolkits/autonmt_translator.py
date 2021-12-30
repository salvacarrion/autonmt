import torch

from autonmt.tasks.translation.bundle.translation_dataset import TranslationDataset
from autonmt.tasks.translation.bundle.vocabulary import BaseVocabulary, Vocabulary
from autonmt.tasks.translation.models import Seq2Seq
from autonmt.tasks.translation.toolkits.base_translator import BaseTranslator
from autonmt.tasks.translation.bundle.search_algorithms import greedy_search, beam_search
from autonmt.utils import *

from typing import Type


class Translator(BaseTranslator):  # AutoNMT Translator

    def __init__(self, model: Type[Seq2Seq], src_vocab=None, trg_vocab=None, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model

        # Translation datasets (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

        # Set vocab (optional)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def _preprocess(self, src_lang, trg_lang, output_path, train_path, val_path, test_path, src_vocab_path,
                    trg_vocab_path, subword_model, **kwargs):

        # Load vocabs (do not replace the ones from the constructor)
        _src_vocab = self.src_vocab if self.src_vocab else self._get_vocab(src_vocab_path + ".vocab", lang=src_lang)
        _trg_vocab = self.trg_vocab if self.trg_vocab else self._get_vocab(trg_vocab_path + ".vocab", lang=trg_lang)

        # Create datasets
        if not kwargs.get("external_data"):  # Training
            self.train_tds = TranslationDataset(file_prefix=train_path, src_lang=src_lang, trg_lang=trg_lang,
                                                src_vocab=_src_vocab, trg_vocab=_trg_vocab)
            self.val_tds = TranslationDataset(file_prefix=val_path, src_lang=src_lang, trg_lang=trg_lang,
                                              src_vocab=_src_vocab, trg_vocab=_trg_vocab)
        else:  # Evaluation
            self.test_tds = TranslationDataset(file_prefix=test_path, src_lang=src_lang, trg_lang=trg_lang,
                                               src_vocab=_src_vocab, trg_vocab=_trg_vocab)

    def _train(self, data_bin_path, checkpoints_path, logs_path, **kwargs):
        # Create and train model
        model = self._get_model(dts=self.train_tds, **kwargs)
        model.fit(self.train_tds, self.val_tds, checkpoints_path=checkpoints_path, logs_path=logs_path, **kwargs)

    def _translate(self, src_lang, trg_lang, beam_width, max_gen_length, batch_size, max_tokens,
                   data_bin_path, output_path, checkpoint_path, model_src_vocab_path, model_trg_vocab_path, **kwargs):
        # Load model
        model = self._get_model(dts=self.test_tds, **kwargs)
        model_state_dict = torch.load(checkpoint_path)
        model.load_state_dict(model_state_dict)

        # Iterative decoding
        search_algorithm = beam_search if beam_width > 1 else greedy_search
        predictions, log_probabilities = search_algorithm(model=model, dataset=self.test_tds,
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

    def _get_vocab(self, vocab, lang):
        if isinstance(vocab, str):
            vocab = Vocabulary(lang=lang).build_from_vocab(filename=vocab)
        elif isinstance(vocab, BaseVocabulary):
            pass
        else:
            raise ValueError("'vocab' must be a path or instance of 'Vocabulary'")

        # Print stuff
        print(f"\t- [INFO]: Loaded '{lang}' vocab with {len(vocab):,} tokens")
        return vocab

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

    @staticmethod
    def _count_model_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params


