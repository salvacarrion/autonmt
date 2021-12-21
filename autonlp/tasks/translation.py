import os.path
from autonlp.utils import *

from autonlp.cmd import fairseq_entry
from autonlp.cmd import tokenizers_entry
from autonlp.datasets.dataset import Dataset
from autonlp.toolkits import Toolkit


TOOLKIT_ENTRY_POINTS = {
    "fairseq": {
        "preprocess": fairseq_entry.preprocess,
        "train": fairseq_entry.train,
        "translate": fairseq_entry.translate,
        "score": fairseq_entry.score,
    },
}


class Translator:

    # Global variables
    total_runs = 0

    def __init__(self, ds, engine="default", force_overwrite=False, interactive=False,
                 run_prefix="model", num_gpus=None, model_path=None, checkpoints_path=None, logs_path=None):
        # Store vars
        self.ds = ds
        self.engine = engine
        self.force_overwrite = force_overwrite
        self.interactive = interactive

        # Parse gpu flag
        self.run_name = f"{run_prefix}_{self.ds.subword_model}_{self.ds.vocab_size}"
        self.num_gpus = None if not num_gpus or num_gpus.strip().lower() == "all" else num_gpus

        # Check if the toolkit engine is supported
        if self.engine in TOOLKIT_ENTRY_POINTS:
            self.fn = TOOLKIT_ENTRY_POINTS[self.engine]
        else:
            raise ValueError(f"Toolkit engine not supported: {self.engine}")

    def preprocess(self):
        # Set vars
        # self.ds = Dataset()
        src_lang = self.ds.src_lang
        trg_lang = self.ds.trg_lang
        output_path = self.ds.get_model_data_bin(toolkit=self.engine)
        train_path = os.path.join(self.ds.get_encoded_path(), self.ds.train_name)
        val_path = os.path.join(self.ds.get_encoded_path(), self.ds.val_name)
        test_path = os.path.join(self.ds.get_encoded_path(), self.ds.test_name)
        src_vocab_path = self.ds.get_src_trg_vocab_path()
        trg_vocab_path = self.ds.get_src_trg_vocab_path()
        self.fn["preprocess"](src_lang=src_lang, trg_lang=trg_lang, output_path=output_path,
                              train_path=train_path, val_path=val_path, test_path=test_path,
                              src_vocab_path=src_vocab_path, trg_vocab_path=trg_vocab_path)

    def train(self):
        self.fn["train"](ds_path=self.ds.get_path(), src_lang=self.ds.src_lang, trg_lang=self.ds.trg_lang,
                         subword_model=self.ds.subword_model, vocab_size=self.ds.vocab_size,
                         force_overwrite=self.force_overwrite, interactive=self.interactive,
                         run_name=self.run_name, model_path=None, num_gpus=self.num_gpus)

    def evaluate(self, eval_datasets, beams, max_gen_length=200, run_name=None, checkpoint_path=None):
        # Set default run_name (if needed)
        if not run_name:
            run_name = f"run_{str(self.total_runs)}"
            print(f"[WARNING]: No 'run_name' was specified. Using last run_name (run_name={run_name})")

        # Checkpoint path
        if not checkpoint_path:
            fname = "checkpoint_best.pt"
            checkpoint_path = os.path.join(self.model_path, self.runs_path, run_name, self.checkpoints_path, fname)

        # Evaluate
        for ds in eval_datasets:  # Evaluation dataset
            # Set evaluation name
            eval_name = "_".join(*ds.id())

            # Check that the datasets are compatible
            assert ds.src_lang == self.ds.src_lang
            assert ds.trg_lang == self.ds.trg_lang

            # [Eval dataset]: Get splits folder
            eval_data_path = os.path.join(ds.base_path, *ds.id(), self.ds.encoded_path, self.ds.subword_model, self.ds.vocab_size)

            # [Trained model]: Create eval folder
            model_data_bin_path = os.path.join(self.model_path, self.runs_path, run_name, self.ds.data_path, self.ds.subword_model, self.ds.vocab_size)
            model_src_vocab_path = os.path.join(model_data_bin_path, "")
            model_trg_vocab_path = os.path.join(model_data_bin_path, "")
            model_eval_path = os.path.join(self.model_path, self.runs_path, run_name, self.eval_path, eval_name)
            model_eval_data_path = os.path.join(model_eval_path, self.eval_data_path)
            model_eval_data_bin_path = os.path.join(model_eval_path, self.eval_data_bin_path)
            make_dir(model_eval_data_path, base_path=self.ds.base_path)

            # [Trained model]: SPM model path
            spm_model_path = os.path.join(self.ds.base_path, *self.ds.id(), self.ds.vocabs_path, self.ds.subword_model, self.ds.vocab_size)
            src_spm_model_path = os.path.join(spm_model_path, f"spm_{self.ds.src_lang}-{self.ds.trg_lang}.model")
            trg_spm_model_path = src_spm_model_path

            # Encode dataset using the SPM of this model
            for fname in [f"{ds.test_name}.{ds.src_lang}", f"{ds.test_name}.{ds.trg_lang}"]:
                data_path = ds.pretokenized_path if ds.pretok_flag else ds.splits_path
                ori_filename = os.path.join(ds.base_path, *ds.id(), data_path, fname)
                new_filename = os.path.join(self.ds.base_path, *self.ds.id(), data_path, fname)

                # Check if the file exists
                if self.force_overwrite or not os.path.exists(new_filename):
                    tokenizers_entry.spm_encode(spm_model_path=spm_model_path, input_file=ori_filename,
                                                output_file=new_filename)

            # Preprocess external data
            if self.force_overwrite or not os.path.exists(model_eval_data_bin_path):
                self.fn["preprocess"](destdir=model_eval_data_bin_path,
                                        train_fname=None, val_fname=None,
                                        test_fname=os.path.join(eval_data_path, self.ds.test_name))
            for beam_width in beams:
                # Create output path (if needed)
                output_path = self.beam_n_path.format(beam_width)
                make_dir(output_path, base_path=self.ds.base_path)

                # Translate
                self.fn["translate"](data_path=model_eval_data_bin_path, checkpoint_path=checkpoint_path,
                                       output_path=output_path,
                                       beam_width=beam_width, max_gen_length=max_gen_length)


    def score(self, eval_datasets, metrics):
        pass
        # # Score
        # toolkit_setup["score_fn"](data_path=eval_data_bin_path, output_path=eval_data_bin_path,
        #                           src_lang=src_lang, trg_lang=trg_lang,
        #                           trg_spm_model_path=spm_model_path, metrics=metrics,
        #                           force_overwrite=force_overwrite, interactive=interactive)