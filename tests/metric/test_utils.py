# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: Utils Tests"""

from kaner.metric.utils import safe_division


def test_safe_division():
    """Test the function `safe_division`."""
    assert safe_division(1, 0) == 0
    assert safe_division(2, 0) == 0
    assert safe_division(2, 0.01) == 2/0.01
    assert safe_division(1, 1) == 1/1
    assert safe_division(1, 2) == 1/2
