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
            print(f"Directory created: {p}")


def get_split_files(split_names, langs):
    return [f"{fname}.{ext}" for fname in split_names for ext in langs]



def get_translation_files(src_lang, trg_lang):
    files = []
    for split in ["train", "val", "test"]:
        for lang in [src_lang, trg_lang]:
            files.append(f"{split}.{lang}")
    return files


def preprocess_text(text):
    try:
        p_whitespace = re.compile(" +")

        # Remove repeated whitespaces "   " => " "
        text = p_whitespace.sub(' ', text)

        # Normalization Form Compatibility Composition
        text = unicodedata.normalize("NFKC", text)

        # Strip whitespace
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


def get_tokens_by_sentence(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        token_sizes = [len(line.strip().split(' ')) for line in lines]
    return token_sizes


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


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(d, savepath):
    with open(savepath, 'w') as f:
        json.dump(d, f)


def create_logger(logs_path, log_level=logging.INFO):
    # Create logget path
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    # Create logger
    mylogger = logging.getLogger()
    mylogger.setLevel(log_level)

    # Define format
    logformat = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Define handlers
    # Log file
    log_handler = logging.FileHandler(os.path.join(logs_path, 'logger.log'), mode='w')
    log_handler.setFormatter(logformat)
    mylogger.addHandler(log_handler)

    # Standard output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logformat)
    mylogger.addHandler(stdout_handler)

    # Print something
    mylogger.info("########## LOGGER STARTED ##########")
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


def parse_sacrebleu(text):
    result = {}
    metrics = json.loads("".join(text))
    metrics = [metrics] if isinstance(metrics, dict) else metrics

    for m_dict in metrics:
        m_name = m_dict['name'].lower().strip()
        result[m_name] = float(m_dict["score"])
    return result


def parse_bertscore(text):
    pattern = r"P: ([01]\.\d*) R: ([01]\.\d*) F1: ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"precision": float(groups[0]), "recall": float(groups[1]), "f1": float(groups[2])}
    return result


def parse_comet(text):
    pattern = r"score: (-?[01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"score": float(groups[0])}
    return result


def parse_beer(text):
    pattern = r"total BEER ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"score": float(groups[0])}
    return result


def count_datasets(datasets):
    counter = 0
    for ds in datasets:  # Training dataset
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:
                counter += 1
    return counter


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
