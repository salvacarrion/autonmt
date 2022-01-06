import autonmt as al
from autonmt.api import *
from autonmt.bundle import *
from autonmt.modules import *
from autonmt.modules.datasets import *
from autonmt.modules.layers import *
from autonmt.modules.models import *
from autonmt.preprocessing import *
from autonmt.search import *
from autonmt.toolkits import *
from autonmt.vocabularies import *


def test_imports():
    # import * only allowed at module level
    assert True
