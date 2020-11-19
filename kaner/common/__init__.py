# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Common utils"""

__all__ = [
    "load_xml_as_json",
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "save_text",
    "load_yaml_as_json",
    "set_seed"
]

import random
import numpy as np
import torch

from .io import (
    load_xml_as_json,
    load_json,
    load_jsonl,
    load_yaml_as_json,
    save_json,
    save_jsonl,
    save_text
)


def set_seed(seed: int) -> None:
    """
    Set seed for deterministic results.

    Args:
        seed (int): Fixed seed.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
