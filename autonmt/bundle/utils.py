import json
import random
import re
from collections import defaultdict
import sys
import logging
import os
import datetime
import time
from pathlib import Path
from collections import Counter

import unicodedata
from tqdm import tqdm


def ask_yes_or_no(question, interactive=True, default=True):
    # Default behaviour when it is not interactive
    if not interactive:
        return default

    # Ask question
    reply = input(f"{question.strip()} [Y/n] ").strip().lower()
    if reply == 'y':
        return True
    else:
        print("Abort.")
        return False


def ask_value(question, interactive=True, default=None):
    # Default behaviour when it is not interactive
    if not interactive:
        return default

    # Ask question
    value = input(f"{question.strip()}").strip().lower()
    return float(value)


def make_dir(path, parents=True, exist_ok=True, base_path=""):
    paths = [path] if isinstance(path, str) else path

    for p in paths:
        p = os.path.join(base_path, p)  # Add base path (if needed)
        if not os.path.exists(p):
            Path(p).mkdir(parents=parents, exist_ok=exist_ok)
            # print(f"Directory created: {p}")


def get_split_files(split_names, langs):
    return [f"{fname}.{ext}" for fname in split_names for ext in langs]


def get_translation_files(src_lang, trg_lang):
    files = []
    for split in ["train", "val", "test"]:
        for lang in [src_lang, trg_lang]:
            files.append(f"{split}.{lang}")
    return files


def preprocess_text(text, letter_case=None, collapse_whitespace=True, strip_whitespace=True, normalization="NFKC",
                    **kwargs):
    try:
        # Set lower/upper case
        if letter_case == "lower":
            text = text.lower()
        elif letter_case == "upper":
            text = text.upper()
        else:
            pass

        # Normalization Form Compatibility Composition
        if normalization:
            text = unicodedata.normalize(normalization.upper(), text)

        # Remove repeated whitespaces "   " => " "
        if collapse_whitespace:
            p_whitespace = re.compile(" +")
            text = p_whitespace.sub(' ', text)

        # Strip whitespace
        if strip_whitespace:
            text = text.strip()

    except TypeError as e:
        # print(f"=> Error preprocessing: '{text}'")
        text = ""
    return text


def preprocess_pairs(src_lines, trg_lines, shuffle):
    assert len(src_lines) == len(trg_lines)

    lines = []
    for _src_line, _trg_line in tqdm(zip(src_lines, trg_lines), total=len(src_lines)):
        src_line = preprocess_text(_src_line)
        trg_line = preprocess_text(_trg_line)

        # Remove empty line
        remove_pair = False
        if len(src_line) == 0 or len(trg_line) == 0:
            remove_pair = True
        # elif math.fabs(len(src)-len(trg)) > 20:
        #     remove_pair = True

        # Add lines
        if not remove_pair:
            lines.append((src_line, trg_line))

    # Shuffle
    if shuffle:
        random.shuffle(lines)

    return lines


def get_frequencies(filename):
    vocab_frequencies = defaultdict(int)
    with open(filename, 'r') as f:
        for line in tqdm(f):
            tokens = line.strip().split(' ')
            for tok in tokens:
                vocab_frequencies[tok] += 1
    return vocab_frequencies


def count_tokens_per_sentence(filename, split_fn=None):
    if split_fn is None:
        split_fn = lambda x: x.strip().split(' ')

    # Count tokens
    with open(filename, 'r') as f:
        token_per_sentence = [len(split_fn(line)) for line in f.readlines()]
    return token_per_sentence



def human_format(num, decimals=2):
    if num < 10000:
        return str(num)
    else:
        magnitude = 0
        template = f'%.{decimals}f%s'

        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0

        return template % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def human_format_int(x, *args, **kwargs):
    return human_format(int(x), decimals=0)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(d, savepath, ignore_empty=True):
    if d or not ignore_empty:
        with open(savepath, 'w') as f:
            json.dump(d, f)
    else:
        print(f"\t- [INFO]: Ignoring empty json. Not saved: {savepath}")


def create_logger(logs_path, log_level=logging.INFO):
    # Create logget path
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    mylogger = logging.getLogger()
    mylogger.setLevel(log_level)

    file_handler = logging.FileHandler(filename=os.path.join(logs_path, "logs.log"), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    mylogger.handlers = [file_handler, stdout_handler]

    # Print something
    mylogger.info("########## LOGGER STARTED ##########")
    mylogger.info(f"- Log level: {str(log_level)}")
    mylogger.info(f"- Logs path: {logs_path}")
    return mylogger


def logged_task(logger, row, fn_name, fn, **kwargs):
    start_fn = time.time()
    start_dt = datetime.datetime.now()
    logger.info(f"***** {fn_name.title()} started *****")
    logger.info(f"----- [{fn_name.title()}] Start time: {start_dt} -----")

    # Call function (...and propagate errors)
    result = None
    # try:
    fn_status = "okay"
    result = fn(**kwargs)
    # except Exception as e:
    #     logger.error(str(e))
    #     fn_status = str(e)

    # Get elapsed time
    end_fn = time.time()
    end_dt = datetime.datetime.now()
    elapsed_fn = end_fn - start_fn
    elapsed_fn_str = str(datetime.timedelta(seconds=elapsed_fn))

    # Log time
    logger.info(f"----- [{fn_name.title()}] Time elapsed (hh:mm:ss.ms): {elapsed_fn_str} -----")
    logger.info(f"----- [{fn_name.title()}] End time: {end_dt} -----")
    logger.info(f"***** {fn_name.title()} ended *****")

    # Store results
    row[f"start_{fn_name}"] = start_dt
    row[f"end_{fn_name}"] = end_dt
    row[f"elapsed_{fn_name}_str"] = elapsed_fn_str
    row[f"{fn_name}_status"] = fn_status

    return result


def parse_split_size(ds_size, max_ds_size):
    # Check size type
    if isinstance(ds_size, tuple):
        return int(min(float(ds_size[0]) * max_ds_size, ds_size[1]))
    elif isinstance(ds_size, float):
        return float(ds_size) * max_ds_size
    elif isinstance(ds_size, int):
        return ds_size
    else:
        raise TypeError("'ds_size' can be a tuple(float, int), float or int")


def read_file_lines(filename, strip=False, remove_break_lines=True):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()] if strip else f.readlines()
        lines = [line.replace('\n', '') for line in lines] if remove_break_lines else lines
    return lines


def write_file_lines(lines, filename, encoding="utf8", strip=False):
    with open(filename, 'w', encoding=encoding.lower()) as f:
        lines = [line.strip() + '\n' for line in lines] if strip else [line + '\n' for line in lines]
        f.writelines(lines)


def replace_in_file(search_string, replace_string, filename, drop_headers=0):
    # Read file
    lines = read_file_lines(filename, strip=False, remove_break_lines=False)

    # Drop headers
    lines = lines[drop_headers:]

    # Clean lines
    lines = [line.replace(search_string, replace_string) for line in lines]

    # Write file
    write_file_lines(lines, filename)


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def build_counter_low_mem(filename, split_fn):
    c = Counter()
    with open(filename, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            tokens = split_fn(line)
            c.update(tokens)
    return c


def norm_counter(c):
    c = Counter(c)
    total = sum(c.values(), 0.0)
    for key in c:
        c[key] /= total
    return c


def parse_json_metrics(text, fields):
    result = {}
    metrics = json.loads("".join(text))
    metrics = [metrics] if isinstance(metrics, dict) else metrics

    for m_dict in metrics:
        m_name = m_dict['name'].lower().strip()  # bleu, chrf,...
        result[m_name] = {}
        for score_name in fields:  # score, precision, recall, f1,...
            result[m_name][score_name] = float(m_dict[score_name])
    return result


def parse_huggingface_json(text):
    return parse_json_metrics(text, fields={"score"})


def parse_huggingface_txt(text):
    raise NotImplementedError("'Huggingface' is only available through the json file")


def parse_sacrebleu_json(text):
    return parse_json_metrics(text, fields={"score"})


def parse_sacrebleu_txt(text):
    raise NotImplementedError("'Sacrebleu' is only available through the json file")


def parse_bertscore_json(text):
    return parse_json_metrics(text, fields={"precision", "recall", "f1"})


def parse_bertscore_txt(text):
    pattern = r"P: ([01]\.\d*) R: ([01]\.\d*) F1: ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"bertscore": {"precision": float(groups[0]), "recall": float(groups[1]), "f1": float(groups[2])}}
    return result


def parse_comet_json(text):
    return parse_json_metrics(text, fields={"score"})


def parse_comet_txt(text):
    pattern = r"score: (-?[01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"comet": {"score": float(groups[0])}}
    return result


def parse_beer_json(text):
    raise NotImplementedError("'Beer' is only available through the text file")


def parse_beer_txt(text):
    pattern = r"total BEER ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"beer": {"score": float(groups[0])}}
    return result


def parse_fairseq_txt(text):
    pattern = r"beam=(\d+): BLEU = (\d+.\d*)"
    line = text[-1].strip()

    groups = re.search(pattern, line).groups()
    result = {"bleu": {"score": float(groups[1])}}
    return result
