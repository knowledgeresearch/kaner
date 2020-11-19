# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: Dataset Tests"""

from kaner.metric.dataset import (
    sentence_distance,
    dataset_diversity
)


def test_sentence_distance():
    """Test the function `sentence_distance`."""
    assert sentence_distance("", "") == 0.0
    assert sentence_distance("ab", "ab") == 0.0
    assert round(sentence_distance("abcd", "ab"), 8) == round(1/6, 8)


def test_dataset_diversity():
    """Test the function `dataset_diversity`."""
    dataset = ["", "ab", "abcd"]
    assert round(dataset_diversity(dataset, 3), 8) == round(13/27, 8)
