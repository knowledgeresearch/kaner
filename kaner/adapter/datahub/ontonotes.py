# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Datahub: OntoNotes"""


import os
import glob
import shutil
import xml.etree.ElementTree as et
from typing import Dict

from kaner.context import GlobalContext as gctx
from kaner.adapter.span import Span, to_spans, to_tags
from .base import BaseDatahub


def _remove_whitespace(s: str):
    """
    Remove whitespaces of a string.

    Args:
        s (str): String to be cleaned.
    """
    chars = []
    for char in s:
        if char != " ":
            chars.append(char)
    return "".join(chars)


def _convert(file_paths: str):
    """
    Preprocess data and convert it to json.

    Args:
        file_paths (str): A list of file path of the original dataset.
    """
    data, labels = [], set()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f_in:
            root = et.ElementTree(et.fromstring(f_in.read())).getroot()
            phrases = [_remove_whitespace(text) for text in root.itertext()]
            entities = [None for _ in range(len(phrases))]
            for child in root.getchildren():
                span_text = _remove_whitespace(child.text)
                span_label = child.attrib["TYPE"]
                index = phrases.index(span_text)
                entities[index] = (span_text, span_label)
            text = ""
            spans = []
            for i, phrase in enumerate(phrases):
                if entities[i] is not None:
                    span = Span(entities[i][1], len(text), len(text) + len(phrase) - 1, phrase, 1.0)
                    spans.append(span)
                    labels.add(entities[i][1])
                text += phrase
            # split document to sentence by '\n'
            tags = to_tags(len(text), spans)
            start, end = 0, 0
            while end <= len(text):
                if end == len(text) or text[end] == "\n":
                    if start != end:
                        new_text = text[start: end]
                        new_tags = tags[start: end]
                        new_spans = to_spans(new_tags, list(new_text), [1.0] * len(new_text))
                        data.append({
                            "text": new_text,
                            "spans": [vars(span) for span in new_spans]
                        })
                    end += 1
                    start = end
                else:
                    end += 1

    return data, list(labels)


@gctx.register_datahub("ontonotes")
class OntoNotes(BaseDatahub):
    """
    Sub-class of BaseDatahub for the dataset OntoNotes. Names are annotated according to the
    following set of types:
        PERSON: People, including fictional
        NORP: Nationalities or religious or polotical groups
        FACILITY: Buildings, airports, highways, bridges, etc.
        ORGANIZATION: Companies, agencies, institutions, etc.
        GPE: Countries, cities, states
        LOCATION: Non-GPE locations, mountain ranges, bodies of water
        PRODUCT: Vehicles, weapons, foods, etc.
        EVENT: Named hurricanes, battles, wars, sports events, etc.
        WORK OF ART: Titles of books, songs, etc.
        LAW: Named documents made into laws
        LANGUAGE: Any named language.
        DATE: Absolute or relative dates or periods
        TIME: Times smaller than a day
        PERCENT: Percentage (including '%')
        MONEY: Monetary values, including unit
        QUANTITY: Measurements, as of weight or distance
        ORDINAL: 'first', 'second'
        CARDINAL: Numerals that do not fall under another type

    References:
        [1] https://catalog.ldc.upenn.edu/LDC2011T03

    Args:
        root_folder (str): The root folder of the dataset.
        task (str): The task of the dataset.
    """

    def __init__(self, root_folder: str, task: str = "NER"):
        super(OntoNotes, self).__init__(
            root_folder, task, "OntoNotes 4.0", ["https://catalog.ldc.upenn.edu/LDC2011T03"]
        )

    def _preprocess(self) -> Dict[str, list]:
        """
        Preprocess the model.
        """
        # unpack the original file
        raw_file_name = "ontonotes-release-4.0_LDC2011T03.tgz"
        raw_folder = os.path.join(self.root_folder, "raw")
        output_folder = os.path.join(raw_folder, "outputs")
        shutil.unpack_archive(
            os.path.join(raw_folder, raw_file_name), output_folder
        )
        file_paths = set()
        zh_folder = os.path.join("ontonotes-release-4.0", "data", "files", "data", "chinese", "annotations")
        for path in os.walk(os.path.join(output_folder, zh_folder)):
            sub_folders = list(path[1]) + ["./"]
            for sub_folder in sub_folders:
                for entity_file_name in glob.glob(os.path.join(path[0], sub_folder, "*.name")):
                    file_paths.add(entity_file_name)
        file_paths = list(file_paths)
        # preprocess
        data, types = _convert(file_paths)
        labels = ["O"]
        for entity_type in types:
            labels.extend(["B-{0}".format(entity_type), "I-{0}".format(entity_type)])
        test_len = int(0.1*len(data))
        train_len = len(data) - 2*test_len
        train_set = data[:train_len]
        dev_set, test_set = data[train_len: train_len+test_len], data[train_len+test_len:]
        results = {
            "data.jsonl": data,
            "train.jsonl": train_set,
            "dev.jsonl": dev_set,
            "test.jsonl": test_set,
            "labels": labels
        }

        return results
