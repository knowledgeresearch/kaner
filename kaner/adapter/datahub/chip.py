# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Datahub: CHIP"""


import os
import shutil
from typing import Dict

from kaner.context import GlobalContext as gctx
from kaner.adapter.span import Span, split_document
from .base import BaseDatahub


def _convert(file_path: str):
    """
    Preprocess data and convert it to json.

    Args:
        file_path (str): The file path of the original dataset.
    """
    data, labels = [], set()
    bad_count = 0
    with open(file_path, "r", encoding="utf-8") as f_in:
        for line in f_in.readlines():
            line = line.replace("\n", "").replace("\r", "")
            if line == "":
                continue
            segs = line.split("|||")
            text = segs[0]
            spans = []
            for seg in segs[1:]:
                seg = seg.strip()
                if seg == "":
                    continue
                boundary = seg.split("    ")
                start, end = int(boundary[0]), int(boundary[1])
                labels.add(boundary[2])
                spans.append(Span(boundary[2], start, end, text[start: end+1], 1.0))
            spans = sorted(spans, key=lambda span: span.start)
            good, bad = split_document(text, spans)
            bad_count += len(bad)
            for sample in good:
                sample["spans"] = [vars(span) for span in sample["spans"]]
                data.append(sample)
    print("bad cutting {0}".format(bad_count))

    return data, list(labels)


@gctx.register_datahub("chip")
class CHIP(BaseDatahub):
    """
    Sub-class of BaseDatahub for the dataset CHIP.

        1	疾病	疾病或综合症|中毒或受伤|器官或细胞受损	dis
        2	临床表现	症状|体征	sym
        3	医疗程序	检查程序|治疗或预防程序	pro
        4	医疗设备	检查设备|治疗设备	equ
        5	药物	药物	dru
        6	医学检验项目	医学检验项目	ite
        7	身体	身体物质|身体部位	bod
        8	科室	科室	dep
        9	微生物类	微生物类	mic

    References:
        [1] https://www.biendata.xyz/competition/chip_2020_1/

    Args:
        root_folder (str): The root folder of the dataset.
        task (str): The task of the dataset.
    """

    def __init__(self, root_folder: str, task: str = "NER"):
        super(CHIP, self).__init__(
            root_folder, task, "CHIP 2020", ["https://www.biendata.xyz/competition/chip_2020_1/"]
        )

    def _preprocess(self) -> Dict[str, list]:
        """
        Preprocess the model.
        """
        # unpack the original file
        raw_file_name = "chip_2020_1_train.zip"
        raw_folder = os.path.join(self.root_folder, "raw")
        output_folder = os.path.join(raw_folder, "outputs")
        shutil.unpack_archive(os.path.join(raw_folder, raw_file_name), output_folder)
        # preprocess
        data, types = _convert(os.path.join(output_folder, "train_data.txt"))
        dev_set, dev_types = _convert(os.path.join(output_folder, "val_data.txt"))
        labels = ["O"]
        for entity_type in set(types + dev_types):
            labels.extend(["B-{0}".format(entity_type), "I-{0}".format(entity_type)])
        train_set = data[:len(data) - len(dev_set)]
        test_set = data[len(data) - len(dev_set):]
        results = {
            "data.jsonl": data + dev_set,
            "train.jsonl": train_set,
            "dev.jsonl": dev_set,
            "test.jsonl": test_set,
            "labels": labels
        }

        return results
