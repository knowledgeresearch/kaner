# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Modelhub: ICD"""

import os
import json
from typing import Dict, Union
import torch

from kaner.context import GlobalContext as gctx
from .base import BaseModelhub


@gctx.register_gazetteer("icd")
class ICD(BaseModelhub):
    """
    International Classification of Diseases (Chinese Version).

    References:
        [1] http://code.nhsa.gov.cn:8000/

    Args:
        root_folder (str): The root folder of the dataset.
    """
    file_names = ["lexicons.txt", "lexicon_configs.json"]

    def __init__(self, root_folder: str):
        super(ICD, self).__init__(root_folder, "ICD", ["http://code.nhsa.gov.cn:8000/"])

    def _preprocess(self) -> Dict[str, Union[list, torch.FloatTensor, dict]]:
        """
        Preprocess the model.
        """
        file_names = ["icd.json", "ccd.json"]
        raw_folder = os.path.join(self.root_folder, "raw")
        # Gazetteer: (lexicon, type, source)
        lexicon_config = {"lexicon_dim": 50}
        lexicons = ["[PAD]\tDiagnosis\tNHSA"]
        for name in file_names:
            with open(os.path.join(raw_folder, name), "r") as f_in:
                for diagnosis in json.loads(f_in.read()):
                    lexicons.append("{0}\t{1}\tNHSA".format(diagnosis["name"], "Diagnosis"))
        lexicon_config["n_lexicons"] = len(lexicons)
        results = {
            "lexicon_configs.json": lexicon_config,
            "lexicons.txt": lexicons
        }

        return results
