# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Modelhub: Gigaword"""

import os
import random
from typing import Dict, Union
import torch

from kaner.context import GlobalContext as gctx
from .base import BaseModelhub


@gctx.register_tokenizer("gigaword")
@gctx.register_gazetteer("gigaword")
class Gigaword(BaseModelhub):
    """
    Chinese Gigaword Fifth Edition was produced by the Linguistic Data Consortium (LDC).
    It is a comprehensive archive of newswire text data that has been acquired from
    Chinese news sources by LDC at the University of Pennsylvania. Chinese Gigaword Fifth
    Edition includes all of the content of the fourth edition of Chinese Gigaword
    (LDC2009T27) plus new data covering the period from January 2009 through December 2010.

    References:
        [1] https://catalog.ldc.upenn.edu/LDC2011T13
        [2] https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view
        [3] https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view
        [4] https://www.tensorflow.org/datasets/catalog/gigaword
        [5] https://github.com/jiesutd/LatticeLSTM

    Args:
        root_folder (str): The root folder of the dataset.
    """

    def __init__(self, root_folder: str):
        super(Gigaword, self).__init__(
            root_folder, "Gigaword",
            [
                "https://github.com/jiesutd/LatticeLSTM",
                "https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view",
                "https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view"
            ]
        )

    def _preprocess(self) -> Dict[str, Union[list, torch.FloatTensor, dict]]:
        """
        Preprocess the model.
        """
        # Token
        token_config = {"token_dim": 50, "token_embedding_type": "static"}
        tokens = ["[PAD]", "[UNK]"]
        token_embeddings = []
        for _ in tokens:
            token_embeddings.append([random.random() for _ in range(token_config["token_dim"])])
        raw_folder = os.path.join(self.root_folder, "raw")
        with open(os.path.join(raw_folder, "gigaword_chn.all.a2b.uni.ite50.vec"), "r") as f_in:
            for line in f_in.readlines():
                line = line.replace("\n", "").replace("\r", "").strip()
                if not line:
                    continue
                elements = line.split(" ")
                tokens.append(elements[0])
                token_embeddings.append([float(elem) for elem in elements[1:]])
        token_config["n_tokens"] = len(tokens)
        token_embeddings = torch.tensor(token_embeddings)

        # Gazetteer: (lexicon, type, source)
        lexicon_config = {"lexicon_dim": 50}
        lexicons = ["[PAD]\tSEGMENTATION\tgigaword"]
        lexicon_embeddings = [[0.0 for _ in range(lexicon_config["lexicon_dim"])]]
        with open(os.path.join(raw_folder, "ctb.50d.vec"), "r") as f_in:
            for line in f_in.readlines():
                line = line.replace("\n", "").replace("\r", "").strip()
                if not line:
                    continue
                elements = line.split(" ")
                lexicons.append("{0}\tSEGMENTATION\tgigaword".format(elements[0]))
                lexicon_embeddings.append([float(elem) for elem in elements[1:]])
        lexicon_config["n_lexicons"] = len(lexicons)
        lexicon_embeddings = torch.tensor(lexicon_embeddings)
        results = {
            "token_configs.json": token_config,
            "tokens.txt": tokens,
            "token_embeddings.checkpoints": token_embeddings,
            "lexicon_configs.json": lexicon_config,
            "lexicons.txt": lexicons,
            "lexicon_embeddings.checkpoints": lexicon_embeddings
        }

        return results
