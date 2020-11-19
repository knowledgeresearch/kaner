# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Output Adapter for Sequence Labeling"""

from typing import List
from kaner.context import GlobalContext as gctx
from .base import BaseOutAdapter


@gctx.register_outadapter("out_seqlab")
class OutSeqlab(BaseOutAdapter):
    """
    OutSeqlab is a sub-class of BaseOutAdapter for the task "Sequence Labeling".

    Args:
        dataset_folder (str): The root folder of a dataset.
        file_name (str): The path of label file.
    """

    def __init__(self, dataset_folder: str, file_name: str, labels: List[str] = None):
        super(OutSeqlab, self).__init__(dataset_folder, file_name, "O", labels)
