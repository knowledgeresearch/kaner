# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Model hubs for preprocessing models to a uniform format"""

__all__ = ["Gigaword", "SGNS", "BERT", "TEC", "ICD"]

from .gigaword import Gigaword
from .sgns import SGNS
from .bert import BERT
from .tec import TEC
from .icd import ICD
