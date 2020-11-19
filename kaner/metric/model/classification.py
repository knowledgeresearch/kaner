# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics: Classification"""

from typing import Tuple

from kaner.metric.utils import safe_division


def compute_prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute precision, recall, f1.
        There are four types of predicted results in Confusion matrix:
            1) true positive (TP) eqv. with hit
            2) true negative (TN) eqv. with correct rejection
            3) false positive (FP) eqv. with false alarm, Type I error
            4) false negative (FN) eqv. with miss, Type II error

    Also we can compute the evaluation metric by the following equations:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2*TP / (2*TP + FP + FN)

    References:
        [1] https://en.wikipedia.org/wiki/Confusion_matrix

    Args:
        tp (int): The number of true positive samples.
        fp (int): The number of false positive samples.
        fn (int): The number of false negative samples.
    """
    precision = safe_division(tp, tp + fp)
    recall = safe_division(tp, tp + fn)
    f1 = safe_division(2 * tp, 2 * tp + fp + fn)

    return (precision, recall, f1)
