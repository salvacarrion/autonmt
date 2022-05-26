import pytorch_lightning as pl
import torch
import wandb
import glob

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from autonmt.bundle.utils import *
from autonmt.modules.datasets.seq2seq_dataset import Seq2SeqDataset
from autonmt.search.beam_search import beam_search
from autonmt.search.greedy_search import greedy_search
from autonmt.toolkits.base import BaseTranslator


import os.path
import shutil
from abc import ABC, abstractmethod
from typing import List, Set

from autonmt.bundle.metrics import *
from autonmt.bundle.utils import *
from autonmt.preprocessing.dataset import Dataset
from autonmt.preprocessing.processors import normalize_file, pretokenize_file, encode_file, decode_file
from autonmt.toolkits.base import _check_datasets, _check_supported_metrics


class AutonmtTranslatorV2(BaseTranslator):  # AutoNMT Translator

    def __init__(self, model, wandb_params=None, print_samples=False, **kwargs):
        super().__init__(engine="autonmt", **kwargs)
        self.model = model
        self.ckpt_cb = None
        self.wandb_params = wandb_params

        # Translation preprocessing (do not confuse with 'train_ds')
        self.train_tds = None
        self.val_tds = None
        self.test_tds = None

        # Filters
        self.filter_train = None
        self.filter_eval = None
        self.filter_train_fn = None
        self.filter_eval_fn = None

        # Other
        self.print_samples = print_samples

    def _preprocess(self, ds, src_lang, trg_lang, output_path, train_path, val_path, test_path,
                    src_vocab_path, trg_vocab_path, force_overwrite, **kwargs):
        # Create preprocessing
        self.subword_model = ds.subword_model
        self.pretok_flag = ds.pretok_flag
        self.src_vocab_path = src_vocab_path
        self.trg_vocab_path = trg_vocab_path

        # Set default values for filter
        self.filter_train = None if not self.filter_train else self.filter_train
        self.filter_eval = [None] if not self.filter_eval else self.filter_eval

        # Set common params
        params = dict(src_lang=src_lang, trg_lang=trg_lang, src_vocab=self.src_vocab, trg_vocab=self.trg_vocab)
        if not kwargs.get("external_data"):  # Training and Validation
            # Training
            self.train_tds = Seq2SeqDataset(file_prefix=train_path, filter_langs=self.filter_train,
                                            filter_fn=self.filter_train_fn, **params, **kwargs)

            # Validation
            self.val_tds = []
            for lang_pairs in self.filter_eval:
                self.val_tds.append(Seq2SeqDataset(file_prefix=val_path, filter_langs=lang_pairs,
                                                   filter_fn=self.filter_eval_fn, **params, **kwargs))
        else:  # Evaluation
            self.test_tds = []
            for lang_pairs in self.filter_eval:
                self.test_tds.append(Seq2SeqDataset(file_prefix=test_path, filter_langs=lang_pairs,
                                                    filter_fn=self.filter_eval_fn, **params, **kwargs))

    def _train(self, data_bin_path, checkpoints_dir, logs_path, max_tokens, batch_size, monitor, run_name,
               num_workers, patience, ds_alias, resume_training, force_overwrite, **kwargs):
        # Notes:
        # - "force_overwrite" is not needed. checkpoints are versioned, not deleted
        # - "resume_training" is not needed. models are initialized by the user.

        # Training dataloaders
        train_loader = DataLoader(self.train_tds, shuffle=True, collate_fn=lambda x: self.train_tds.collate_fn(x, max_tokens=max_tokens), batch_size=batch_size, num_workers=num_workers)

        # Validation dataloaders
        val_loaders = []
        for val_tds_i in self.val_tds:
            val_loaders.append(DataLoader(val_tds_i, shuffle=False, collate_fn=lambda x: val_tds_i.collate_fn(x, max_tokens=max_tokens), batch_size=batch_size, num_workers=num_workers))
        if len(val_loaders) == 1 and "dataloader_idx_" in monitor:  # Check
            raise ValueError(f"[MONITOR] You don't need to specify the 'dataloader_idx' ({monitor}) in the monitor when there is just one validation loader")

        # Additional information for metrics
        self.model._src_vocab = self.train_tds.src_vocab
        self.model._trg_vocab = self.train_tds.trg_vocab
        self.model._subword_model = self.subword_model
        self.model._pretok_flag = self.pretok_flag
        self.model._src_model_vocab_path = self.src_vocab_path
        self.model._trg_model_vocab_path = self.trg_vocab_path
        self.model._filter_train = self.filter_train
        self.model._filter_eval = self.filter_eval
        self.model._print_samples = self.print_samples

        # Callbacks: Checkpoint
        callbacks = []
        mode = "min" if "loss" in monitor else "max"
        filename = "checkpoint_best__{epoch}-{" + monitor.replace('/', '-') + ":.2f}"
        self.ckpt_cb = ModelCheckpoint(dirpath=checkpoints_dir, save_top_k=1, monitor=monitor, mode=mode,
                                       filename=filename, save_weights_only=True)
        self.ckpt_cb.FILE_EXTENSION = ".pt"
        callbacks += [self.ckpt_cb]

        # Callback: EarlyStop
        if patience:
            early_stop = EarlyStopping(monitor=monitor, patience=patience, mode=mode)
            callbacks += [early_stop]

        # Loggers
        tb_logger = TensorBoardLogger(save_dir=logs_path, name=run_name)
        loggers = [tb_logger]

        # Add wandb logger (if requested)
        if self.wandb_params:
            wandb_logger = WandbLogger(name=f"{ds_alias}_{run_name}", **self.wandb_params)
            loggers.append(wandb_logger)

            # Monitor
            wandb_logger.watch(self.model)

        # Training
        remove_params = {'weight_decay', 'criterion', 'optimizer', 'patience', 'seed', 'learning_rate', "fairseq_args"}
        pl_params = {k: v for k, v in kwargs.items() if k not in remove_params}
        trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **pl_params)  # pl_params must be compatible with PL
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loaders)

        # Force finish
        if self.wandb_params:
            wandb.finish()

    def _translate(self, src_lang, trg_lang, beam_width, max_len_a, max_len_b, batch_size, max_tokens,
                   data_bin_path, output_path, load_best_checkpoint, num_workers, devices, accelerator,
                   model_ds, model_src_vocab_path, model_trg_vocab_path, force_overwrite, checkpoints_dir=None, **kwargs):
        # Checkpoint
        if load_best_checkpoint:
            if self.ckpt_cb:
                checkpoint_path = self.ckpt_cb.best_model_path
            else:
                # Find last checkpoint
                checkpoints = sorted(glob.iglob(os.path.join(checkpoints_dir, "*.pt")), key=os.path.getctime, reverse=True)
                if not checkpoints:
                    raise ValueError(f"No checkpoints were found in {checkpoints_dir}")
                elif len(checkpoints) > 1:
                    checkpoint_path = checkpoints[0]
                    print(f"[WARNING] Multiple checkpoints were found. Using latest: {checkpoint_path}")
                else:
                    checkpoint_path = checkpoints[0]
                    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

            # Load checkpoint
            model_state_dict = torch.load(checkpoint_path)
            model_state_dict = model_state_dict.get("state_dict", model_state_dict)
            self.model.load_state_dict(model_state_dict)

        # Set evaluation model
        if accelerator in {"auto", "cuda", "gpu"} and self.model.device.type != "cuda":
            print(f"\t-[INFO]: Setting 'cuda' as the model's device")
            self.model = self.model.cuda()

        # Iterative decoding
        search_algorithm = beam_search if beam_width > 1 else greedy_search
        for i, t_ds in enumerate(self.test_tds):
            new_output_path = os.path.join(output_path, f"{i}")
            tok_flag = [os.path.exists(os.path.join(new_output_path, f)) for f in ["hyp.tok", "ref.tok", "src.tok"]]
            if force_overwrite or not all(tok_flag):
                predictions, log_probabilities = search_algorithm(model=self.model, dataset=t_ds,
                                                                  sos_id=t_ds.src_vocab.sos_id,
                                                                  eos_id=t_ds.src_vocab.eos_id,
                                                                  batch_size=batch_size, max_tokens=max_tokens,
                                                                  beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                                                  num_workers=num_workers)
                # Decode output
                make_dir([new_output_path])
                self._postprocess_output(predictions=predictions, output_path=new_output_path,
                                         src_lines=t_ds.src_lines, ref_lines=t_ds.trg_lines)

            # Postprocess tokenized files
            for fname, lang in [("hyp", model_ds.trg_lang), ("ref", model_ds.trg_lang), ("src", model_ds.src_lang)]:
                input_file = os.path.join(new_output_path, f"{fname}.tok")
                output_file = os.path.join(new_output_path, f"{fname}.txt")
                model_vocab_path = model_src_vocab_path if lang == model_ds.src_lang else model_trg_vocab_path

                # Post-process files
                decode_file(input_file=input_file, output_file=output_file, lang=lang,
                            subword_model=model_ds.subword_model, pretok_flag=model_ds.pretok_flag,
                            model_vocab_path=model_vocab_path, remove_unk_hyphen=True,
                            force_overwrite=force_overwrite)

    def _postprocess_output(self, predictions, output_path, src_lines=None, ref_lines=None):
        """
        Important: src and ref will NOT be overwritten with the original preprocessed files to deal with filtering.
        This may lead to problems with unknowns
        """
        # Decode: hyp
        hyp_tok = [self.test_tds[0].trg_vocab.decode(tokens) for tokens in predictions]  # Vocab is shared across tds
        write_file_lines(lines=hyp_tok, filename=os.path.join(output_path, "hyp.tok"), insert_break_line=True)

        # Decode: ref
        if ref_lines:
            write_file_lines(lines=ref_lines, filename=os.path.join(output_path, "ref.tok"), insert_break_line=True)

        # Decode: src
        if src_lines:
            write_file_lines(lines=src_lines, filename=os.path.join(output_path, "src.tok"), insert_break_line=True)

    def translate(self, model_ds, eval_ds, beams, max_len_a, max_len_b, truncate_at,
                  batch_size, max_tokens, num_workers, force_overwrite, **kwargs):
        print(f"=> [Translate v2]: Started. ({model_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Set run names
        run_name = model_ds.get_run_name(self.run_prefix)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Checkpoints dir
        checkpoints_dir = model_ds.get_model_checkpoints_path(self.engine, run_name)

        # [Trained model]: Create eval folder
        model_src_vocab_path = model_ds.get_vocab_file(lang=model_ds.src_lang)  # Needed to preprocess
        model_trg_vocab_path = model_ds.get_vocab_file(lang=model_ds.trg_lang)  # Needed to preprocess
        model_eval_data_path = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name,
                                                                 eval_name=eval_name)
        model_eval_data_bin_path = model_ds.get_model_eval_data_bin_path(toolkit=self.engine, run_name=run_name,
                                                                         eval_name=eval_name)

        # Create dirs
        make_dir([model_eval_data_path, model_eval_data_bin_path])

        # [Encode extern data]: Encode test data using the subword model of the trained model
        for ts_fname in [fname for fname in eval_ds.split_names_lang if eval_ds.test_name in fname]:
            lang = ts_fname.split('.')[-1]
            input_file = eval_ds.get_split_path(ts_fname)  # as raw as possible
            output_file = model_ds.get_model_eval_data_path(toolkit=self.engine, run_name=run_name,
                                                            eval_name=eval_name)

            # Create directories
            make_dir([
                os.path.join(output_file, "raw"),
                os.path.join(output_file, "normalized"),
                os.path.join(output_file, "tokenized"),
                os.path.join(output_file, "encoded"),
            ])

            # Copy raw
            raw_file = os.path.join(output_file, "raw", ts_fname)
            shutil.copyfile(input_file, raw_file)
            input_file = raw_file

            # Normalize data
            norm_file = os.path.join(output_file, "normalized", ts_fname)
            normalize_file(input_file=input_file, output_file=norm_file,
                           normalizer=model_ds.normalizer, force_overwrite=force_overwrite)
            input_file = norm_file

            # Pretokenize data (if needed)
            if model_ds.pretok_flag:
                pretok_file = os.path.join(output_file, "tokenized", ts_fname)
                pretokenize_file(input_file=input_file, output_file=pretok_file, lang=lang,
                                 force_overwrite=force_overwrite)
                input_file = pretok_file

            # Encode file
            enc_file = os.path.join(output_file, "encoded", ts_fname)
            encode_file(ds=model_ds, input_file=input_file, output_file=enc_file,
                        lang=lang, merge_vocabs=model_ds.merge_vocabs, truncate_at=truncate_at,
                        force_overwrite=force_overwrite)

        # Preprocess external data
        test_path = os.path.join(model_eval_data_path, "encoded", eval_ds.test_name)
        self._preprocess(ds=model_ds, src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                         output_path=model_eval_data_bin_path,
                         train_path=None, val_path=None, test_path=test_path,
                         src_vocab_path=model_src_vocab_path, trg_vocab_path=model_trg_vocab_path,
                         subword_model=model_ds.subword_model, pretok_flag=model_ds.pretok_flag,
                         external_data=True, force_overwrite=force_overwrite,
                         **kwargs)

        # Iterate over beams
        for beam in beams:
            start_time = time.time()
            # Create output path (if needed)
            output_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name,
                                                       beam=beam)
            make_dir(output_path)
            self._translate(
                src_lang=model_ds.src_lang, trg_lang=model_ds.trg_lang,
                beam_width=beam, max_len_a=max_len_a, max_len_b=max_len_b, batch_size=batch_size,
                max_tokens=max_tokens,
                data_bin_path=model_eval_data_bin_path, output_path=output_path, checkpoints_dir=checkpoints_dir,
                model_src_vocab_path=model_src_vocab_path, model_trg_vocab_path=model_trg_vocab_path,
                num_workers=num_workers, model_ds=model_ds, force_overwrite=force_overwrite,
                **kwargs)

    @staticmethod
    def _count_model_parameters(model):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params


    def score(self, model_ds: Dataset, eval_ds: Dataset, beams: List[int], metrics: Set[str], force_overwrite, **kwargs):
        print(f"=> [Score]: Started. ({model_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Set run names
        run_name = model_ds.get_run_name(self.run_prefix)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Iterate over beams
        for beam in beams:
            for i, t_ds in enumerate(self.test_tds):
                start_time = time.time()

                # Paths
                beam_path = model_ds.get_model_beam_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=f"{beam}/{i}")
                scores_path = model_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name, beam=f"{beam}/{i}")

                # Create dirs
                make_dir([scores_path])

                # Set input files (results)
                src_file_path = os.path.join(beam_path, "src.txt")
                ref_file_path = os.path.join(beam_path, "ref.txt")
                hyp_file_path = os.path.join(beam_path, "hyp.txt")

                # Check that the paths exists
                if not all([os.path.exists(p) for p in [src_file_path, ref_file_path, hyp_file_path]]):
                    raise IOError("Missing files to compute scores")

                # Score: bleu, chrf and ter
                if self.TOOL2METRICS["sacrebleu"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"sacrebleu_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_sacrebleu(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, metrics=metrics)

                # Score: bertscore
                if self.TOOL2METRICS["bertscore"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"bertscore_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_bertscore(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file, trg_lang=model_ds.trg_lang)

                # Score: comet
                if self.TOOL2METRICS["comet"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"comet_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_comet(src_file=src_file_path, ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file)

                 # Score: fairseq
                if self.TOOL2METRICS["fairseq"].intersection(metrics):
                    output_file = os.path.join(scores_path, f"fairseq_scores.txt")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_fairseq(ref_file=ref_file_path, hyp_file=hyp_file_path, output_file=output_file)

                # Huggingface metrics
                hg_metrics = {x[3:] for x in metrics if x.startswith("hg_")}
                if hg_metrics:
                    output_file = os.path.join(scores_path, f"huggingface_scores.json")
                    if force_overwrite or not os.path.exists(output_file):
                        compute_huggingface(src_file=src_file_path, hyp_file=hyp_file_path, ref_file=ref_file_path,
                                            output_file=output_file, metrics=hg_metrics, trg_lang=model_ds.trg_lang)

                print(f"\t- [INFO]: Scoring time (beam={str(beam)}): {str(datetime.timedelta(seconds=time.time() - start_time))}")


    def parse_metrics(self, model_ds, eval_ds, beams: List[int], metrics: Set[str], force_overwrite, **kwargs):
        print(f"=> [Parsing]: Started. ({model_ds.id2(as_path=True)})")

        # Check preprocessing
        _check_datasets(train_ds=model_ds, eval_ds=eval_ds)
        assert model_ds.dataset_lang_pair == eval_ds.dataset_lang_pair

        # Check supported metrics
        metrics_valid = _check_supported_metrics(metrics, self.METRICS2TOOL.keys())
        if not metrics_valid:
            return

        # Metrics to retrieve
        metric_tools = self._get_metrics_tool(metrics)

        # Set run names
        run_name = model_ds.get_run_name(self.run_prefix)
        eval_name = '_'.join(eval_ds.id())  # Subword model and vocab size don't characterize the dataset!

        # Walk through beams
        scores = {
            "engine": kwargs.get("engine"),
            "lang_pair": model_ds.dataset_lang_pair,
            "train_dataset": model_ds.dataset_name,
            "eval_dataset": eval_ds.dataset_name,
            "subword_model": str(model_ds.subword_model).lower(),
            "vocab_size": str(model_ds.vocab_size).lower(),
            "run_name": run_name,
            "train_max_lines": model_ds.dataset_lines,
            "beams": {},
            "config": self.config,
        }

        # Iterate over beams
        for beam in beams:
            for i, t_ds in enumerate(self.test_tds):
                # Paths
                scores_path = model_ds.get_model_scores_path(toolkit=self.engine, run_name=run_name, eval_name=eval_name,
                                                             beam=f"{beam}/{i}")

                # Walk through metric files
                beam_scores = {}
                for m_tool in metric_tools:
                    values = self.TOOL_PARSERS[m_tool]
                    m_parser, ext = values["py"]
                    m_fname = f"{values['filename']}.{ext}"

                    # Read file
                    filename = os.path.join(scores_path, m_fname)
                    if os.path.exists(filename):
                        try:
                            with open(filename, 'r') as f:
                                m_scores = m_parser(text=f.readlines())
                                for m_name, m_values in m_scores.items():  # [bleu_score, chrf_score, ter_score], [bertscore_precision]
                                    for score_name, score_value in m_values.items():
                                        m_name_full = f"{m_tool}_{m_name}_{score_name}".lower().strip()
                                        beam_scores[m_name_full] = score_value
                        except Exception as e:
                            print(f"\t- [PARSING ERROR]: ({m_fname}) {str(e)}")
                    else:
                        print(f"\t- [WARNING]: There are no metrics from '{m_tool}'")

                # Add beam scores
                scores["beams"].update({f"beam{str(beam)}-{i}": beam_scores})
        return scores
