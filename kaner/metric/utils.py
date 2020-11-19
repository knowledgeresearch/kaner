# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: Utils"""

from typing import Union


def safe_division(numerator: Union[int, float], denominator: Union[int, float]) -> Union[int, float]:
    """
    Safely division without considering the denominator is zero.

    Args:
        numerator (Union[int, float]): The numerator.
        denominator (Union[int, float]): The denominator.
    """
    return numerator / denominator if denominator else 0.0
