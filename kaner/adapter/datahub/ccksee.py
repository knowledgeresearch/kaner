# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Datahub: CCKSEE"""

import os
import re
import shutil
from copy import deepcopy
from typing import Dict, List
from collections import defaultdict
import json

from kaner.context import GlobalContext as gctx
from kaner.adapter.span import Span
from kaner.adapter.event import Event
from .base import BaseDatahub


def split_and_cut(data: List[dict], max_seq_len: int = 510, max_query_len: int = 128, back_offset: int = 32, test_pp: float = 0.1) -> tuple:
    """
    Split dataset into train/dev/test set, and cut long document into short pharases.

    Args:
        data (List[dict]): The original dataset containing all datapoints.
        max_seq_len (int): The maximum length of text.
        max_query_len (int): The maximum length of query.
        back_offset (int): Back offset when sliding window.
        test_pp (float): The proportion of the test set.
    """
    assert isinstance(test_pp, float) and 0.0 < test_pp < 0.5
    trainlen, testlen = len(data) - int(len(data) * test_pp)*2, int(len(data) * test_pp)
    trainset, devset, testset = data[:trainlen], data[trainlen: trainlen + testlen], data[trainlen + testlen:]
    max_seq_len -= max_query_len

    def sliding_window(partition: List[dict]) -> List[dict]:
        """
        Cut long document into small pieces.
        """
        new_partition = []
        for i, datapoint in enumerate(partition):
            text = datapoint["text"]
            spans = []
            for event in datapoint["events"]:
                event_type = event["event_type"]
                for argument in event["arguments"]:
                    role_name, role_value = argument["label"], argument["text"]
                    label = "{0}.{1}".format(event_type, role_name)
                    indices = [m.start() for m in re.finditer(re.escape(role_value), text)]
                    for idx in indices:
                        spans.append(vars(Span(label, idx, idx + len(role_value) - 1, role_value, 1.0)))
            # cut document
            pointer = 0
            seg_id = 0
            while pointer < len(text):
                new_text = text[pointer: pointer + max_seq_len]
                new_spans = []
                for span in spans:
                    if span["start"] >= pointer and span["end"] < pointer + max_seq_len:
                        new_span = deepcopy(span)
                        new_span["start"] -= pointer
                        new_span["end"] -= pointer
                        new_spans.append(new_span)
                new_partition.append({"id": "Doc{0}-Part{1}".format(i, seg_id), "text": new_text, "spans": new_spans})
                seg_id += 1
                pointer += max_seq_len - back_offset
        # check
        for datapoint in new_partition:
            for span in datapoint["spans"]:
                assert span["text"] == datapoint["text"][span["start"]: span["end"] + 1]

        return new_partition

    trainset = sliding_window(trainset)
    devset = sliding_window(devset)
    testset = sliding_window(testset)

    return (trainset, devset, testset)


@gctx.register_datahub("ccksee")
class CCKSEE(BaseDatahub):
    """
    Sub-class of BaseDatahub for the dataset CCKSEE.

    References:
        [1] https://www.biendata.xyz/competition/ccks_2020_4_2/data/

    Args:
        root_folder (str): The root folder of the dataset.
        task (str): The task of the dataset.
    """
    file_names = ["datarf.jsonl", "data.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl", "labels"]

    def __init__(self, root_folder: str, task: str = "EE"):
        super(CCKSEE, self).__init__(
            root_folder, task, "CCKSEE", ["https://www.biendata.xyz/competition/ccks_2020_4_2/data/"]
        )

    def _preprocess(self) -> Dict[str, list]:
        """
        Preprocess the model.
        """
        # unpack the original file
        raw_file_name = "ccks4_2_Data.zip"
        raw_folder = os.path.join(self.root_folder, "raw")
        output_folder = os.path.join(raw_folder, "outputs")
        shutil.unpack_archive(os.path.join(raw_folder, raw_file_name), output_folder)
        file_path = os.path.join(output_folder, "ccks 4_2 Data", "event_element_train_data_label.txt")
        # preprocess
        special_tokens = ["&nbsp;", "<br>", "&quot;", "&gt;", "&lt;", "&amp;", "&apos;"]
        raw_data = []
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin.readlines():
                line = line.replace("\n", "").replace("\t", "").strip()
                if line == "":
                    continue
                raw_data.append(json.loads(line))
        data, raw_event_schemes = [], defaultdict(set)
        for raw_datapoint in raw_data:
            events = []
            for raw_event in raw_datapoint["events"]:
                event_type = raw_event["event_type"]
                raw_event.pop("event_type")
                raw_event.pop("event_id")
                empty = []
                for key, value in raw_event.items():
                    if value == "" or key == "":
                        empty.append(key)
                        continue
                    for stok in special_tokens:
                        raw_event[key] = raw_event[key].replace(stok, "")
                for key in empty:
                    raw_event.pop(key)
                events.append(Event(event_type, 1.0, [Span(role, -1, -1, value, 1.0) for role, value in raw_event.items()]))
                for role_name in raw_event.keys():
                    raw_event_schemes[event_type].add(role_name)
            for stok in special_tokens:
                raw_datapoint["content"] = raw_datapoint["content"].replace(stok, "")
            data.append({"text": raw_datapoint["content"], "events": [vars(event) for event in events]})
        labels = ["O"]
        for event_type, roles in raw_event_schemes.items():
            for role in roles:
                labels.extend(["B-{0}.{1}".format(event_type, role), "I-{0}.{1}".format(event_type, role)])
        # cut long document and split dataset
        trainset, devset, testset = split_and_cut(data)

        results = {
            "datarf.jsonl": data,  # data with raw format (events)
            "data.jsonl": trainset + devset + testset,
            "train.jsonl": trainset,
            "dev.jsonl": devset,
            "test.jsonl": testset,
            "labels": labels
        }

        return results
