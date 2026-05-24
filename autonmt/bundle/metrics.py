import os.path

from tqdm import tqdm

import bert_score
import sacrebleu

from autonmt.bundle import utils
from autonmt.bundle.logger import get_logger

log = get_logger(__name__)

# HuggingFace migrated metrics out of ``datasets`` into the standalone ``evaluate``
# package (``load_metric`` was removed). We support both for backwards compatibility:
# if neither is available, ``compute_huggingface`` raises at call time, not at import.
try:
    from evaluate import load as _hf_load_metric
except ImportError:
    try:
        from datasets import load_metric as _hf_load_metric  # legacy datasets<2.20
    except ImportError:
        _hf_load_metric = None


def compute_sacrebleu(ref_file, hyp_file, output_file, metrics):
    if not metrics:
        return

    # Read files
    hyp_lines = utils.read_file_lines(hyp_file, autoclean=True)
    ref_lines = utils.read_file_lines(ref_file, autoclean=True)
    assert len(ref_lines) == len(hyp_lines)

    # Check if files have content
    if not hyp_lines or not ref_lines:
        raise ValueError("Files empty (hyp/ref)")

    # Compute scores
    scores = _sacrebleu(hyp_lines, ref_lines, metrics)

    # Save json
    utils.save_json(scores, output_file)


def _sacrebleu(hyp_lines, ref_lines, metrics, trg_lang="", tokenize=None):
    scores = []
    if "bleu" in metrics:
        # Score
        bleu = sacrebleu.metrics.BLEU(trg_lang=trg_lang, tokenize=tokenize)  # Tokenizer automatically
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
    return scores


def compute_bertscore(ref_file, hyp_file, output_file, trg_lang):
    # Read file
    ref_lines = utils.read_file_lines(ref_file, autoclean=True)
    hyp_lines = utils.read_file_lines(hyp_file, autoclean=True)
    assert len(ref_lines) == len(hyp_lines)

    # Check if files have content
    if not hyp_lines or not ref_lines:
        raise ValueError("Files empty (hyp/ref)")

    # Compute scores
    scores = _bertscore(hyp_lines, ref_lines, trg_lang)

    # Save json
    utils.save_json(scores, output_file)


def _bertscore(hyp_lines, ref_lines, lang):
    # Score
    precision, recall, f1 = bert_score.score(hyp_lines, ref_lines, lang=lang)

    scores = [
        {"name": "bertscore",
         "precision": float(precision.mean()),
         "recall": float(recall.mean()),
         "f1": float(f1.mean()),
         }
    ]
    return scores


def compute_comet(src_file, ref_file, hyp_file, output_file):
    # Read file
    src_lines = utils.read_file_lines(src_file, autoclean=True)
    ref_lines = utils.read_file_lines(ref_file, autoclean=True)
    hyp_lines = utils.read_file_lines(hyp_file, autoclean=True)
    assert len(ref_lines) == len(hyp_lines) == len(src_lines)

    # Check if files have content
    if not hyp_lines or not ref_lines or not src_lines:
        raise ValueError("Files empty (hyp/ref/src)")

    # Compute scores
    scores = _comet(src_lines, hyp_lines, ref_lines)

    # Save json
    utils.save_json(scores, output_file)


def _comet(src_lines, hyp_lines, ref_lines):
    try:
        import comet
    except ImportError as e:
        raise ImportError(
            "'unbabel-comet' is required to compute COMET scores. "
            "Install with: pip install unbabel-comet"
        ) from e

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
    return scores


def compute_fairseq(ref_file, hyp_file, output_file):
    # Get generate-tests
    generate_test_path = os.path.join(os.path.dirname(hyp_file), "generate-test.txt")
    if os.path.exists(generate_test_path):
        # Read, parse and save lines
        lines = [utils.read_file_lines(generate_test_path, autoclean=True)[-1]]
        utils.write_file_lines(lines=lines, filename=output_file, insert_break_line=True)
    else:
        log.info("\t- [INFO]: No 'generate-test.txt' was found.")


def compute_huggingface(src_file, hyp_file, ref_file, output_file, metrics, trg_lang):
    scores = []

    if not metrics:
        return

    if _hf_load_metric is None:
        raise ImportError(
            "HuggingFace metrics require the 'evaluate' package (or legacy 'datasets'<2.20). "
            "Install with: pip install evaluate"
        )

    # Read files
    hyp_lines = utils.read_file_lines(hyp_file, autoclean=True)
    ref_lines = utils.read_file_lines(ref_file, autoclean=True)
    assert len(ref_lines) == len(hyp_lines)

    # Load metric
    for metric in metrics:
        try:
            # Tokenize sentences
            hyp_lines_tok = [x for x in hyp_lines]
            ref_lines_tok = [[x] for x in ref_lines]

            # Compute score
            hg_metric = _hf_load_metric(metric)
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
            log.info(f"\t- [HUGGINGFACE ERROR]: Ignoring metric: {str(metric)}.\n"
                  f"\t                       Message: {str(e)}")

    # Save json
    utils.save_json(scores, output_file)



