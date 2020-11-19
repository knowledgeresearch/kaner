# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics: Named Entity Recognition"""

from typing import Tuple

from kaner.adapter.span import to_spans
from kaner.metric.utils import safe_division


def compute_ner_prf1(pred_batch_tags: list, gold_batch_tags: list) -> Tuple[float, float, float]:
    """
    Compute precision, recall, f1 for named entity recognition.

    Args:
        pred_batch_tags (list): A batch of sentence tags from predictions.
        gold_batch_tags (list): A batch of sentence tags from gold annotations.
    """
    num_sentences = len(pred_batch_tags)
    true_pred, pred_samples, gold_samples = [], [], []
    for i in range(num_sentences):
        pred = [str(span) for span in to_spans(pred_batch_tags[i], ["X"]*len(pred_batch_tags[i]), [1.0]*len(pred_batch_tags[i]))]
        gold = [str(span) for span in to_spans(gold_batch_tags[i], ["X"]*len(gold_batch_tags[i]), [1.0]*len(gold_batch_tags[i]))]
        pred_samples.extend(pred)
        gold_samples.extend(gold)
        true_pred.extend(list(set(pred).intersection(set(gold))))
    precision = safe_division(len(true_pred), len(pred_samples))
    recall = safe_division(len(true_pred), len(gold_samples))
    f1 = safe_division(2*precision*recall, precision+recall)

    return (precision, recall, f1)
