# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Datahub: MSRA NER"""


import os
import shutil
from typing import Dict

from kaner.context import GlobalContext as gctx
from kaner.adapter.span import Span
from kaner.common import load_xml_as_json
from .base import BaseDatahub


def _convert(file_path: str):
    """
    Preprocess data and convert it to json.

    Args:
        file_path (str): The file path of the original dataset.
    """
    data, labels = [], []
    content = load_xml_as_json("gb18030", file_path)
    for sentence in content["TEXT"]["SENTENCE"]:
        spans = []
        words = []
        length = 0
        for word in sentence["w"]:
            if isinstance(word, str):
                words.append(word)
                length += len(word)
            elif isinstance(word, dict):
                key = list(word.keys())[0]
                sub_spans = []
                if isinstance(word[key], dict):
                    sub_spans.append(word[key])
                elif isinstance(word[key], list):
                    sub_spans = word[key]
                for sub_word in sub_spans:
                    words.append(sub_word["#text"])
                    label = sub_word["@TYPE"]
                    spans.append(
                        Span(label, length, length+len(words[-1])-1, words[-1], 1.0)
                    )
                    labels.append(label)
                    length += len(words[-1])
            elif word is None:
                continue
            else:
                raise ValueError("type: {0}, {1}".format(type(word), word))
        data.append({
            "text": "".join(words),
            "spans": [vars(span) for span in spans]
        })

    return data, labels


@gctx.register_datahub("msraner")
class MSRANER(BaseDatahub):
    """
    Sub-class of BaseDatahub for the dataset MSRA NER.
        This dataset contains the following entity types:
            1) NAMEX: Person, Location, Orgnization
            2) TIMEX: Date, Duration, Time
            3) NUMEX: Percent, Money, Frequency, Integer, Fraction, Decimal, Ordinal, Rate
            4) MEASUREX: Age, Weight, Length, Temperature, Area, Capacity, Speed, Acceration, Other measures
            5) ADDREX: Email, Phone, Fax, Telex, WWW, Postalcode

    References:
        [1] https://www.microsoft.com/en-us/download/details.aspx?id=52531

    Args:
        root_folder (str): The root folder of the dataset.
        task (str): The task of the dataset.
    """

    def __init__(self, root_folder: str, task: str = "NER"):
        super(MSRANER, self).__init__(
            root_folder, task, "MSRA NER", ["https://www.microsoft.com/en-us/download/details.aspx?id=52531"]
        )

    def _preprocess(self) -> Dict[str, list]:
        """
        Preprocess the model.
        """
        # unpack the original file
        raw_file_name = "msra-chinese-word-segmentation-data-v1.zip"
        raw_folder = os.path.join(self.root_folder, "raw")
        output_folder = os.path.join(raw_folder, "outputs")
        shutil.unpack_archive(
            os.path.join(raw_folder, raw_file_name), output_folder
        )
        # preprocess
        labels = ["O"]
        train_dev_set, train_dev_labels = _convert(os.path.join(output_folder, "msra_bakeoff3_training.xml"))
        test_set, test_labels = _convert(os.path.join(output_folder, "msra_bakeoff3_test.xml"))
        dev_len = int(len(train_dev_set) * 0.1)
        train_set = train_dev_set[: len(train_dev_set) - dev_len]
        dev_set = train_dev_set[len(train_dev_set) - dev_len:]
        for label in set(train_dev_labels + test_labels):
            labels.extend(["B-{0}".format(label), "I-{0}".format(label)])
        results = {
            "data.jsonl": train_dev_set + test_set,
            "train.jsonl": train_set,
            "dev.jsonl": dev_set,
            "test.jsonl": test_set,
            "labels": labels
        }

        return results
