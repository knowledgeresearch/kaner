# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Modelhub: TEC"""

import os
from typing import Dict, Union
import torch

from kaner.context import GlobalContext as gctx
from .base import BaseModelhub


@gctx.register_gazetteer("tec")
class TEC(BaseModelhub):
    """
    Taobao ECommece Gazetteer, including brand names and product names.

    References:
        [1] https://github.com/PhantomGrapes/MultiDigraphNER/tree/master/data/dics

    Args:
        root_folder (str): The root folder of the dataset.
    """
    file_names = ["lexicons.txt", "lexicon_configs.json"]

    def __init__(self, root_folder: str):
        super(TEC, self).__init__(root_folder, "TEC", ["https://github.com/PhantomGrapes/MultiDigraphNER/tree/master/data/dics"])

    def _preprocess(self) -> Dict[str, Union[list, torch.FloatTensor, dict]]:
        """
        Preprocess the model.
        """
        file_names = {
            "BRAND": ["brand1.dic", "brand2.dic", "brand3.dic", "brand4.dic"],
            "PRODUCT": ["product1.dic", "product2.dic", "product3.dic"]
        }
        raw_folder = os.path.join(self.root_folder, "raw")
        # Gazetteer: (lexicon, type, source)
        lexicon_config = {"lexicon_dim": 50}
        lexicons = ["[PAD]\tSEGMENTATION\ttaobao"]
        for lexicon_type, names in file_names.items():
            for name in names:
                with open(os.path.join(raw_folder, name), "r") as f_in:
                    for line in f_in.readlines():
                        line = line.replace("\n", "").replace("\r", "")
                        lexicons.append("{0}\t{1}\ttaobao".format(line, lexicon_type))
        lexicon_config["n_lexicons"] = len(lexicons)
        results = {
            "lexicon_configs.json": lexicon_config,
            "lexicons.txt": lexicons
        }

        return results
