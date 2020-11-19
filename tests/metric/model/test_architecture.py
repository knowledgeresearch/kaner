# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: Architecture Tests"""

import torch.nn as nn

from kaner.metric.model.architecture import count_parameters


def test_count_parameters():
    """Test the function `count_parameters`."""
    assert count_parameters(nn.Linear(10, 20, True)) == 10*20 + 20
    assert count_parameters(nn.Linear(10, 20, False)) == 10*20
