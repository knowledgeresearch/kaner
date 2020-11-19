# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Output Adapter for Sequence Labeling based on MRC mode"""

from kaner.context import GlobalContext as gctx
from .seqlab import OutSeqlab


@gctx.register_outadapter("out_mrc")
class OutMRC(OutSeqlab):
    """
    OutMRC is a sub-class of OutSeqlab for the task "Machine Reading Comprehension".
    """

    def __init__(self):
        super(OutSeqlab, self).__init__(None, None, "O", ["O", "B-Span", "I-Span"])
