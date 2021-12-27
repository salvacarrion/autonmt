import subprocess
from sacremoses import MosesTokenizer, MosesDetokenizer
import sentencepiece as spm

from autonmt import utils
from autonmt.cmd.cmd_tokenizers import *
from autonmt.cmd.cmd_metrics import *

from tqdm import tqdm

import sacrebleu
import bert_score
import comet
from datasets import load_metric


def moses_tokenizer(input_file, output_file, lang, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_moses_tokenizer(input_file, output_file, lang, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_moses_tokenizer(input_file, output_file, lang)


def py_moses_tokenizer(input_file, output_file, lang):
    # Read lines
    lines = utils.read_file_lines(input_file)

    # Tokenize
    mt = MosesTokenizer(lang=lang)
    lines = [mt.tokenize(line, return_str=True) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file)


def moses_detokenizer(lang, input_file, output_file, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_moses_detokenizer(lang, input_file, output_file, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_moses_detokenizer(lang, input_file, output_file)


def py_moses_detokenizer(lang, input_file, output_file):
    # Read lines
    lines = utils.read_file_lines(input_file)

    # Detokenize
    mt = MosesDetokenizer(lang=lang)
    lines = [mt.detokenize(line, return_str=True) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file)


def spm_encode(spm_model_path, input_file, output_file, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_spm_encode(spm_model_path, input_file, output_file, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_spm_encode(spm_model_path, input_file, output_file)


def py_spm_encode(spm_model_path, input_file, output_file):
    # Read lines
    lines = utils.read_file_lines(input_file)

    # Encode
    s = spm.SentencePieceProcessor(model_file=spm_model_path)
    lines = [' '.join(s.encode(line, out_type=str)) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file)


def spm_decode(spm_model_path, input_file, output_file, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_spm_decode(spm_model_path, input_file, output_file, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_spm_decode(spm_model_path, input_file, output_file)


def py_spm_decode(spm_model_path, input_file, output_file):
    # Read lines
    lines = utils.read_file_lines(input_file)

    # Decode
    s = spm.SentencePieceProcessor(model_file=spm_model_path)
    lines = [s.decode_pieces(line.split(' ')) for line in tqdm(lines, total=len(lines))]

    # Save file
    utils.write_file_lines(lines=lines, filename=output_file)


def spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size)


def py_spm_train(input_file, model_prefix, subword_model, vocab_size, input_sentence_size):
    # Train model
    spm.SentencePieceTrainer.train(input=input_file, model_prefix=model_prefix,
                                   model_type=subword_model, vocab_size=vocab_size,
                                   input_sentence_size=input_sentence_size,
                                   pad_id=3)


def compute_huggingface(src_file, hyp_file, ref_file, output_file, metrics, trg_lang, use_cmd, conda_env_name):
    if not metrics:
        return

    if use_cmd:
        print("\t- [INFO]: No cmd interface for 'huggingface'. Using Python package.")
        # raise NotImplementedError("compute_huggingface")
    py_huggingface(src_file, hyp_file, ref_file, output_file, metrics, trg_lang)


def py_huggingface(src_file, hyp_file, ref_file, output_file, metrics, trg_lang):
    scores = []

    # Read files
    hyp_lines = utils.read_file_lines(hyp_file)
    ref_lines = utils.read_file_lines(ref_file)
    assert len(ref_lines) == len(hyp_lines)

    # Load metric
    for metric in metrics:
        try:
            # Tokenize sentences
            hyp_lines_tok = [x for x in hyp_lines]
            ref_lines_tok = [[x] for x in ref_lines]

            # Compute score
            hg_metric = load_metric(metric)
            hg_metric.add_batch(predictions=hyp_lines_tok, references=ref_lines_tok)
            result = hg_metric.compute()

            # Format results
            d = {
                "name": metric,
            }
            d.update(result)

            # Add results
            scores.append(d)
        except Exception as e:
            print(f"\t- [HUGGINGFACE ERROR]: Ignoring metric: {str(metric)}.\n"
                  f"\t                       Message: {str(e)}")

    # Save json
    utils.save_json(scores, output_file)


def compute_sacrebleu(ref_file, hyp_file, output_file, metrics, use_cmd, conda_env_name):
    if not metrics:
        return

    if use_cmd:
        cmd = cmd_sacrebleu(ref_file, hyp_file, output_file, metrics, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_sacrebleu(ref_file, hyp_file, output_file, metrics)


def py_sacrebleu(ref_file, hyp_file, output_file, metrics, **kwargs):
    # Read files
    hyp_lines = utils.read_file_lines(hyp_file)
    ref_lines = utils.read_file_lines(ref_file)
    assert len(ref_lines) == len(hyp_lines)

    # Check if files have content
    if not hyp_lines or not ref_lines:
        raise ValueError("Files empty (hyp/ref)")

    scores = []
    if "bleu" in metrics:
        bleu = sacrebleu.metrics.BLEU()
        d = bleu.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(bleu.get_signature())
        scores.append(d)

    if "chrf" in metrics:
        chrf = sacrebleu.metrics.CHRF()
        d = chrf.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(chrf.get_signature())
        scores.append(d)

    if "ter" in metrics:
        ter = sacrebleu.metrics.TER()
        d = ter.corpus_score(hyp_lines, [ref_lines]).__dict__
        d["signature"] = str(ter.get_signature())
        scores.append(d)

    # Save json
    utils.save_json(scores, output_file)


def compute_bertscore(ref_file, hyp_file, output_file, trg_lang, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_bertscore(ref_file, hyp_file, output_file, trg_lang, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_bertscore(ref_file, hyp_file, output_file, trg_lang)


def py_bertscore(ref_file, hyp_file, output_file, trg_lang):
    # Read file
    ref_lines = utils.read_file_lines(ref_file)
    hyp_lines = utils.read_file_lines(hyp_file)
    assert len(ref_lines) == len(hyp_lines)

    # Check if files have content
    if not hyp_lines or not ref_lines:
        raise ValueError("Files empty (hyp/ref)")
    
    # Score
    precision, recall, f1 = bert_score.score(hyp_lines, ref_lines, lang=trg_lang)

    scores = [
        {"name": "bertscore",
         "precision": float(precision.mean()),
         "recall": float(recall.mean()),
         "f1": float(f1.mean()),
         }
    ]

    # Save json
    utils.save_json(scores, output_file)


def compute_comet(src_file, ref_file, hyp_file, output_file, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_cometscore(src_file, ref_file, hyp_file, output_file, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        py_comet(src_file, ref_file, hyp_file, output_file)


def py_comet(src_file, ref_file, hyp_file, output_file):
    # Read file
    src_lines = utils.read_file_lines(src_file)
    ref_lines = utils.read_file_lines(ref_file)
    hyp_lines = utils.read_file_lines(hyp_file)
    assert len(ref_lines) == len(hyp_lines) == len(src_lines)

    # Check if files have content
    if not hyp_lines or not ref_lines or not src_file:
        raise ValueError("Files empty (hyp/ref/src)")

    # Get model
    model_path = comet.download_model("wmt20-comet-da")
    model = comet.load_from_checkpoint(model_path)

    # Score
    data = {"src": src_lines, "mt": hyp_lines, "ref": ref_lines}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    seg_scores, sys_score = model.predict(data)

    scores = [
        {"name": "comet",
         # "seg_scores": seg_scores,
         "score": sys_score,
         }
    ]

    # Save json
    utils.save_json(scores, output_file)


def compute_beer(ref_file, hyp_file, output_file, use_cmd, conda_env_name):
    if use_cmd:
        cmd = cmd_beer(ref_file, hyp_file, output_file, conda_env_name)
        print(f"\t- [INFO]: Command used: {cmd}")
    else:
        print("\t- [INFO]: No python interface for 'Beer'. Command-line version only ('use_cmd=True').")
