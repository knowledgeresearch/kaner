# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Base datahub"""

import os
from typing import List, Dict

from kaner.common import save_jsonl, save_text


class BaseDatahub:
    """
    BaseDatahub defines the basic interfaces of Datahub, which provides preprocessing
    of a specific dataset.

    Args:
        root_folder (str): The root folder of the dataset.
        task (str): The task of the dataset.
        dataset_name (str): The name of the dataset to be preprocessed.
        urls (List[str]): This list is regarded as tips for downloading datasets.
    """
    file_names = ["data.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl", "labels"]

    def __init__(self, root_folder: str, task: str, dataset_name: str, urls: List[str]):
        super(BaseDatahub, self).__init__()
        self.root_folder = root_folder
        self.task = task
        self.dataset_name = dataset_name
        self.urls = urls

    def preprocess(self, re_preprocess: bool = False):
        """
        Preprocess the dataset.

        Args:
            re_preprocess (bool): If it is True, the dataset will be preprocessed no matter whether it has been preprocessed
                or not.
        """
        if not os.path.isdir(self.root_folder):
            os.makedirs(self.root_folder)
        is_preprocessed = len(set(os.listdir(self.root_folder)).intersection(set(self.file_names))) == len(self.file_names)
        if re_preprocess or not is_preprocessed:
            # check whether raw files exists
            raw_folder = os.path.join(self.root_folder, "raw")
            if not os.path.isdir(raw_folder) or len(os.listdir(raw_folder)) == 0:
                if not os.path.isdir(raw_folder):
                    os.makedirs(raw_folder)
                print("\033[33mWarning: {0} is not initialized!\033[0m".format(self.dataset_name))
                print("You need to manually download the dataset {0} into the folder {1}".format(self.dataset_name, raw_folder))
                for url in self.urls:
                    print(url)
                exit(0)

            # preprocess raw data with a unified format
            results = self._preprocess()
            assert all([key in self.file_names for key in results.keys()]),\
                   "Preprocessed files should be included in {0}".format(self.file_names)
            for key in results.keys():
                if key.endswith(".jsonl"):
                    assert isinstance(results[key], list)
                    save_jsonl(results[key], self.root_folder, key)
                else:
                    assert isinstance(results[key], list)
                    save_text("\n".join(results[key]), self.root_folder, key)
            print("Datahub has been preprocessed successfully in {0} with {1}".format(self.root_folder, results.keys()))

    def _preprocess(self) -> Dict[str, list]:
        """
        Preprocess the model.
        """
        raise NotImplementedError
