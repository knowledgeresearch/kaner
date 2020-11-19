# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: Classification Tests"""

from kaner.metric.model.classification import compute_prf1


def test_compute_prf1():
    """Test the function `compute_prf1`."""
    assert compute_prf1(40, 20, 10) == (40/(40+20), 40/(40+10), 2*40/(2*40+20+10))
    assert compute_prf1(25, 15, 25) == (25/(25+15), 25/(25+25), 2*25/(2*25+15+25))
