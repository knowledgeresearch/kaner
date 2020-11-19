# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Datahub: Weibo NER"""


import os
from typing import Dict

from kaner.context import GlobalContext as gctx
from kaner.adapter.span import to_spans
from .base import BaseDatahub


def _convert(file_path: str):
    """
    Preprocess data and convert it to json.

    Args:
        file_path (str): The file path of the original dataset.
    """
    data, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f_in:
        lines = [line.replace("\n", "").replace("\r", "") for line in f_in.readlines()]
        start = 0
        while start < len(lines):
            end = start
            while end < len(lines) and lines[end] != "":
                end += 1
            if start == end:
                start += 1
            else:
                tokens = []
                tags = []
                for elem in lines[start: end]:
                    tokens.append(elem[0])
                    tags.append(elem.split("\t")[1].split(".")[0])
                spans = to_spans(tags, tokens, [1.0]*len(tokens))
                data.append({
                    "text": "".join(tokens),
                    "spans": [vars(span) for span in spans]
                })
                for span in spans:
                    labels.append(span.label)
                start = end

    return data, labels


@gctx.register_datahub("weiboner")
class WeiboNER(BaseDatahub):
    """
    Sub-class of BaseDatahub for the dataset Weibo NER.

    References:
        [1] https://github.com/hltcoe/golden-horse

    Args:
        root_folder (str): The root folder of the dataset.
        task (str): The task of the dataset.
    """

    def __init__(self, root_folder: str, task: str = "NER"):
        super(WeiboNER, self).__init__(
            root_folder, task, "Weibo NER", ["https://github.com/hltcoe/golden-horse"]
        )

    def _preprocess(self) -> Dict[str, list]:
        """
        Preprocess the model.
        """
        raw_folder = os.path.join(self.root_folder, "raw")
        # preprocess
        labels = ["O"]
        train_set, train_labels = _convert(os.path.join(raw_folder, "weiboNER_2nd_conll.train"))
        dev_set, dev_labels = _convert(os.path.join(raw_folder, "weiboNER_2nd_conll.dev"))
        test_set, test_labels = _convert(os.path.join(raw_folder, "weiboNER_2nd_conll.test"))
        for label in set(train_labels + dev_labels + test_labels):
            labels.extend(["B-{0}".format(label), "I-{0}".format(label)])
        results = {
            "data.jsonl": train_set + dev_set + test_set,
            "train.jsonl": train_set,
            "dev.jsonl": dev_set,
            "test.jsonl": test_set,
            "labels": labels
        }

        return results
