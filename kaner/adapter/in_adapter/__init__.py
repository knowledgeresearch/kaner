# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Input Adapters"""

__all__ = [
    "split_dataset",
    "BaseInAdapter",
    "InSeqlab",
    "InSES",
    "InCGN",
    "InMDGG"
]


from typing import Tuple

from kaner.common import load_jsonl
from .base import BaseInAdapter
from .seqlab import InSeqlab
from .ses import InSES
from .cgn import InCGN
from .mdgg import InMDGG


def split_dataset(
        folder: str, test_pp: float = 0.1, dataset_pp: float = 1.0, resplit: bool = False
) -> Tuple[list, list, list]:
    """
    Split dataset into three parts: train/dev/test.

    Args:
        folder: The folder of dataset. (*.jsonl)
        test_pp: The proportion of the test set.
        dataset_pp: The proportion of used samples in the dataset. This option is
            usually used for you to debug your model. It can accelerate the speed
            of the workflow of the model training or testing.
        resplit: Whether to re-split the dataset.
    """
    assert 0. < test_pp < 0.5 and 0. < test_pp <= 1.0
    assert 0. < dataset_pp <= 1.0
    if resplit:
        data = load_jsonl("utf-8", folder, "data.jsonl")
        num_test = int(test_pp*len(data))
        num_train = len(data) - num_test*2
        train = data[:num_train]
        dev, test = data[num_train: num_train+num_test], data[num_train+num_test:]
    else:
        train = load_jsonl("utf-8", folder, "train.jsonl")
        dev = load_jsonl("utf-8", folder, "dev.jsonl")
        test = load_jsonl("utf-8", folder, "test.jsonl")
    if dataset_pp < 1.0:
        train = train[:int(len(train) * dataset_pp)]
        dev = dev[:int(len(dev) * dataset_pp)]
        test = test[:int(len(test) * dataset_pp)]
        print("NOTE: dataset_pp = {0}!".format(dataset_pp))
    print("[Dataset: {0}] {1} train, {2} dev, {3} test. (resplit: {4})".format(
        folder, len(train), len(dev), len(test), resplit)
    )

    return (train, dev, test)
