# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Modelhub: SGNS"""

import os
import shutil
import bz2
from typing import Dict, Union
import torch

from kaner.context import GlobalContext as gctx
from .base import BaseModelhub


@gctx.register_gazetteer("sgns")
class SGNS(BaseModelhub):
    """
    Chinese Word Vectors 中文词向量

    References:
        [1] https://github.com/Embedding/Chinese-Word-Vectors
        [2] https://github.com/DianboWork/Graph4CNER

    Args:
        root_folder (str): The root folder of the dataset.
    """
    file_names = [
        "lexicons.txt", "lexicon_embeddings.checkpoints", "lexicon_configs.json"
    ]

    def __init__(self, root_folder: str):
        super(SGNS, self).__init__(root_folder, "SGNS", ["https://github.com/DianboWork/Graph4CNER"])

    def _preprocess(self) -> Dict[str, Union[list, torch.FloatTensor, dict]]:
        """
        Preprocess the model.
        """
        # unpack the original file
        raw_file_name = "sgns.merge.word.bz2"
        raw_folder = os.path.join(self.root_folder, "raw")
        output_folder = os.path.join(raw_folder, "outputs")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # https://stackoverflow.com/questions/49073295/possible-to-decompress-bz2-in-python-to-a-file-instead-of-memory
        file_path = os.path.join(output_folder, "sgns")
        if not os.path.isfile(file_path):
            with bz2.BZ2File(os.path.join(raw_folder, raw_file_name)) as fr, open(file_path, "wb") as fw:
                shutil.copyfileobj(fr, fw, 1000000)
        # Gazetteer: (lexicon, type, source)
        lexicon_config = {"lexicon_dim": 300, "token_embedding_type": "static"}
        lexicons = ["[PAD]\tSEGMENTATION\tsgns"]
        lexicon_embeddings = [[0.0 for _ in range(lexicon_config["lexicon_dim"])]]
        with open(file_path, "r") as f_in:
            line = f_in.readline()
            while True:
                line = f_in.readline().replace("\n", "").replace("\r", "").strip()
                if not line:
                    break
                elements = line.split(" ")
                lexicons.append("{0}\tSEGMENTATION\tsgns".format(elements[0]))
                lexicon_embeddings.append([float(elem) for elem in elements[1:]])
        lexicon_config["n_lexicons"] = len(lexicons)
        lexicon_embeddings = torch.tensor(lexicon_embeddings)
        results = {
            "lexicon_configs.json": lexicon_config,
            "lexicons.txt": lexicons,
            "lexicon_embeddings.checkpoints": lexicon_embeddings
        }

        return results
