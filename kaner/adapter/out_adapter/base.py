# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Base Output Adapter"""

import os
from typing import List

from kaner.common import save_text


class BaseOutAdapter:
    """
    BaseOutAdapter defines the basic interfaces for the data post-processing.

    Args:
        dataset_folder: The root folder of a dataset.
        file_name: The path of label file.
        unk_label: Unknown label, such as 'O' in sequence labeling.
        labels (List[str]): Label list from the memory. If this parameter is not None, then the output adapter
            will load labels from memory only.
    """

    def __init__(self, dataset_folder: str, file_name: str, unk_label: str, labels: List[str] = None):
        super(BaseOutAdapter, self).__init__()
        self._dataset_folder = dataset_folder
        self._file_name = file_name
        self._unk_label = unk_label
        self._to_ids = {}
        self._to_tags = []
        self._load(labels)

    def _load(self, labels: List[str]):
        """
        Load labels from file or memory.

        Args:
            labels (List[str]): Label list from the memory. If this parameter is not None, then the output adapter
                will load labels from memory only.
        """
        if labels is None:
            label_path = os.path.join(self._dataset_folder, self._file_name)
            with open(label_path, "r") as f_in:
                labels = [line.replace("\n", "") for line in f_in.readlines()]
        for label in labels:
            self._to_ids[label] = len(self._to_tags)
            self._to_tags.append(label)

    def save(self, folder: str, file_name: str = None):
        """
        Save labels to a folder.

        Args:
            folder (str): The folder where labels will save into.
            file_name (str): The file name of labels.
        """
        if file_name is None:
            file_name = self._file_name
        save_text("\n".join(self._to_tags), folder, file_name)

    @property
    def unk_label(self):
        """
        Unknown label, such as 'O' in sequence labeling.
        """
        return self._unk_label

    @property
    def unk_id(self):
        """
        ID of Unknown label, such as 'O' in sequence labeling.
        """
        return self[self._unk_label]

    def __len__(self):
        """
        Return the number of labels.
        """
        return len(self._to_tags)

    def __getitem__(self, item):
        """
        Return the corresponding map value: id->label, label->id.
        """
        assert isinstance(item, (int, str))
        if isinstance(item, int):
            if 0 <= item < len(self):
                return self._to_tags[item]
            return self.unk_label
        if item in self._to_ids.keys():
            return self._to_ids[item]
        return self.unk_id

    def convert_labels_to_ids(self, labels: list):
        """
        Convert a list of label to the corresponding ids.

        Args:
            labels (list): A list of label to be converted.
        """
        return [self[label] for label in labels]
