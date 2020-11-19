# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Output Adapters"""

__all__ = [
    "BaseOutAdapter",
    "OutSeqlab",
    "OutMRC"
]


from .base import BaseOutAdapter
from .seqlab import OutSeqlab
from .mrc import OutMRC
