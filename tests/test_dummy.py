import os
import pathlib
import shutil

import autonmt as al

from autonmt.modules.models.transfomer import Transformer
from autonmt.toolkits.autonmt import AutonmtTranslator
from autonmt.toolkits.fairseq import FairseqTranslator
from autonmt.preprocessing.builder import DatasetBuilder
from autonmt.bundle.report import generate_report, summarize_scores

import pytest


def test_dummy():
    assert True
