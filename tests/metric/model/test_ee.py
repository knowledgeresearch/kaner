# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: EE Tests"""

from kaner.metric.model.ee import compute_ee_prf1


def test_compute_ee_prf1():
    """Test the function `compute_ee_prf1`."""
    pred_tags = [
        [["O", "B-E1", "I-E1", "O"], ["O", "B-E2", "I-E2", "O"]],
        [["O", "O"], ["O", "B-E2"]]
    ]
    gold_tags = [
        [["O", "B-E1", "I-E1", "O"], ["O", "B-E2", "I-E2", "O"]],
        [["O", "O"], ["O", "B-E2"]]
    ]
    assert compute_ee_prf1(pred_tags, gold_tags) == (1.0, 1.0, 1.0)
    pred_tags = [
        [["O", "B-E1", "I-E1", "O"], ["O", "B-E2", "I-E2", "O"]],
        [["O", "O"], ["O", "B-E2"]]
    ]
    gold_tags = [
        [["O", "B-E1", "I-E1", "O"], ["O", "B-E2", "O", "O"]],
        [["B-E1", "O"], ["O", "B-E2"]]
    ]
    assert [round(n, 2) for n in compute_ee_prf1(pred_tags, gold_tags)] == [round(2/3, 2), 0.5, round(4/7, 2)]
