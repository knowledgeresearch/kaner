# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics: Architecture"""

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of parameters in a model.

    Reference:
        [1] https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model

    Args:
        model (nn.Module): The model we want to calculate.
    """
    assert isinstance(model, nn.Module)
    total_params = sum(p.numel() for p in model.parameters())

    return total_params
