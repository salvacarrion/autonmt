from autonlp.cmd import fairseq_entry

TOOLKIT_ENTRY_POINTS = {
    "fairseq": {
        "preprocess": fairseq_entry.preprocess,
        "train": fairseq_entry.train,
        "translate": fairseq_entry.translate,
        "score": fairseq_entry.score,
    },
}


class Toolkit:

    def __init__(self, engine,
                 src_lang, trg_lang,
                 src_vocab_path, trg_vocab_path,
                 src_spm_model_path,  trg_spm_model_path,
                 force_overwrite=False, interactive=False):
        # Store vars
        self.engine = engine
        self.force_overwrite = force_overwrite
        self.interactive = interactive



    def preprocess(self, *args, **kwargs):
        self.fn["preprocess"](*args, **kwargs)

    def train(self, *args, **kwargs):
        self.fn["train"](*args, **kwargs)

    def translate(self, *args, **kwargs):
        self.fn["translate"](*args, **kwargs)

    def score(self, *args, **kwargs):
        self.fn["score"](*args, **kwargs)
