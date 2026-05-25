"""File, directory, JSON and text-encoding helpers.

Single place for low-level I/O so the rest of the package never reaches for
``os``/``shutil``/``json`` directly. Keep this module free of dataset/plot/metric
knowledge — it is consumed by everything.
"""
import json
import os
import shutil
import time
from pathlib import Path

from autonmt.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def make_dir(path, parents=True, exist_ok=True, base_path=""):
    paths = [path] if isinstance(path, str) else path
    for p in paths:
        p = os.path.join(base_path, p)
        if not os.path.exists(p):
            Path(p).mkdir(parents=parents, exist_ok=exist_ok)


def is_dir_empty(path):
    return os.listdir(path) == []


def empty_dir(path, safe_seconds=0):
    if safe_seconds > 0:
        log.info(f"\t- Deleting files... (safe mode: ON | waiting {safe_seconds} seconds)")
        time.sleep(safe_seconds)
    else:
        log.info(f"\t- Deleting files... (safe mode: OFF | no wait)")
    shutil.rmtree(path)
    make_dir(path)


def rename_file(base_path, old_name, new_name):
    try:
        os.rename(os.path.join(base_path, old_name), os.path.join(base_path, new_name))
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Text files
# ---------------------------------------------------------------------------

def clean_file_line(line, encoding="utf8"):
    line = line.decode(encoding, errors="replace") if isinstance(line, bytes) else line
    line = line.replace('\r', '')
    line = line.strip()
    return line


def read_file_lines(filename, autoclean=False, remove_empty=False, encoding="utf8"):
    with open(filename, 'rb') as f:  # Sometimes there are byte characters
        lines = []
        for line in f.readlines():
            if autoclean:
                line = clean_file_line(line, encoding)
            else:
                line = line.decode(encoding.lower(), errors="replace")
            if not remove_empty or line:
                lines.append(line)
    return lines


def write_file_lines(lines, filename, autoclean=False, insert_break_line=False, encoding="utf8"):
    tail = '\n' if insert_break_line else ''
    with open(filename, 'w', encoding=encoding.lower()) as f:
        lines = [(clean_file_line(line) if autoclean else line) + tail for line in lines]
        f.writelines(lines)


def replace_in_file(search_string, replace_string, filename, drop_headers=0):
    lines = read_file_lines(filename, autoclean=False)
    lines = lines[drop_headers:]
    lines = [line.replace(search_string, replace_string) for line in lines]
    write_file_lines(lines=lines, filename=filename, insert_break_line=False)


def count_file_lines(file_path):
    return sum(1 for _ in open(file_path, 'rb'))


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(d, savepath, ignore_empty=True, allow_overwrite=True):
    if d or not ignore_empty:
        if allow_overwrite or not os.path.exists(savepath):
            with open(savepath, 'w') as f:
                json.dump(d, f)
    else:
        log.info(f"\t- [INFO]: Ignoring empty json. Not saved: {savepath}")


# ---------------------------------------------------------------------------
# Byte-fallback encoding helpers (used by the ``bytes`` tokenizer)
# ---------------------------------------------------------------------------

def text2hex(text, return_str=False):
    """Encode each Unicode code point as its hexadecimal value.

    We use ``hex(ord(c))`` (code-point view) rather than UTF-8 byte view because
    the byte-fallback tokenizer operates on Unicode scalars, not on raw bytes.
    """
    hex_values = [hex(ord(c)) for c in text]
    return ' '.join(hex_values) if return_str else hex_values


def hex2text(hex_values, return_str=False):
    if isinstance(hex_values, str):
        hex_values = hex_values.split(' ')
    elif not isinstance(hex_values, list):
        raise ValueError("hex_values must be a list of strings or a string")
    text_values = [chr(int(c, 16)) for c in hex_values]
    return ' '.join(text_values) if return_str else text_values


# ---------------------------------------------------------------------------
# Tiny CLI helper (the only ``input()`` call in the codebase)
# ---------------------------------------------------------------------------

def ask_yes_or_no(question, interactive=True, default=True):
    if not interactive:
        return default
    reply = input(f"{question.strip()} [Y/n] ").strip().lower()
    if reply == 'y':
        return True
    log.info("Abort.")
    return False
