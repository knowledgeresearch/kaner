# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: NER Tests"""

from kaner.metric.model.ner import compute_ner_prf1


def test_compute_ner_prf1():
    """Test the function `compute_ner_prf1`."""
    pred_tags = [
        ["O", "B-X", "I-X", "O"],
        ["O", "B-Y"]
    ]
    gold_tags = [
        ["O", "B-X", "I-X", "O"],
        ["O", "B-Y"]
    ]
    assert compute_ner_prf1(pred_tags, gold_tags) == (1.0, 1.0, 1.0)
    pred_tags = [
        ["O", "B-X", "I-X", "O"],
        ["O", "B-Y", "O"]
    ]
    gold_tags = [
        ["O", "B-X", "I-X", "O"],
        ["O", "B-Y", "I-Y"]
    ]
    assert compute_ner_prf1(pred_tags, gold_tags) == (0.5, 0.5, 0.5)
