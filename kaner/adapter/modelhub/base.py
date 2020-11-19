# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Base modelhub"""

import os
from typing import Dict, Union, List
from collections import OrderedDict
import torch
from kaner.common import save_json, save_text


class BaseModelhub:
    """
    BaseModelhub defines the basic interfaces of Modelhub, which provides preprocessing
    of a specific pretrained model, such as embeddings, pretrained language models.

    Args:
        root_folder (str): The root folder of the model.
        dataset_name (str): The name of the dataset to be preprocessed.
        urls (List[str]): This list is regarded as tips for downloading datasets.
    """
    file_names = [
        "tokens.txt", "token_embeddings.checkpoints", "token_configs.json",
        "lexicons.txt", "lexicon_embeddings.checkpoints", "lexicon_configs.json"
    ]

    def __init__(self, root_folder: str, dataset_name: str, urls: List[str]):
        super(BaseModelhub, self).__init__()
        self.root_folder = root_folder
        self.dataset_name = dataset_name
        self.urls = urls

    def preprocess(self, re_preprocess: bool = False):
        """
        Preprocess the model with error handling.

        Args:
            re_preprocess (bool): If it is True, the model will be preprocessed no matter whether it has been preprocessed
                or not.
        """
        is_preprocessed = len(set(os.listdir(self.root_folder)).intersection(set(self.file_names))) == len(self.file_names)
        if re_preprocess or not is_preprocessed:
            # check whether raw files exists
            raw_folder = os.path.join(self.root_folder, "raw")
            if not os.path.isdir(raw_folder) or len(os.listdir(raw_folder)) == 0:
                if not os.path.isdir(raw_folder):
                    os.makedirs(raw_folder)
                print("\033[33mWarning: {0} is not initialized!\033[0m".format(self.dataset_name))
                print("You need to manually download the model {0} into the folder {1}".format(self.dataset_name, raw_folder))
                for url in self.urls:
                    print(url)
                exit(0)

            # preprocess raw data with a unified format
            results = self._preprocess()
            assert all([key in self.file_names for key in results.keys()]),\
                   "Preprocessed files should be included in {0}".format(self.file_names)
            for key in results.keys():
                if key.endswith(".txt"):
                    assert isinstance(results[key], list)
                    save_text("\n".join(results[key]), self.root_folder, key)
                elif key.endswith(".checkpoints"):
                    assert isinstance(results[key], (torch.FloatTensor, OrderedDict))
                    torch.save(results[key], os.path.join(self.root_folder, key))
                elif key.endswith(".json"):
                    assert isinstance(results[key], dict)
                    save_json(results[key], self.root_folder, key)
            print("Modelhub has been preprocessed successfully in {0} with {1}".format(self.root_folder, results.keys()))

    def _preprocess(self) -> Dict[str, Union[list, torch.FloatTensor, dict]]:
        """
        Preprocess the model.
        """
        raise NotImplementedError
