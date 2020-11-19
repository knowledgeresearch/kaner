# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics: Dataset"""

from typing import Union, List
from kaner.metric.utils import safe_division


def sentence_distance(s1: str, s2: str, n: int = 3) -> float:
    r"""
    Given a pair of sentences, s1 and s2, calculate the reverse of the mean Jaccard Index between the
    sentences' n-grams sets to represent the semantic distances.

        DIST(a, b) = 1 - \sum_{i=1}^{n} \frac{ngram_{a} \cap ngram_{b}}{ngram_{a} \cup ngram_{b}}

    Args:
        s1 (str): Sentence 1.
        s2 (str): Sentence 2.
        n (int): The maximum n-gram length.

    Reference:
        [1] https://www.aclweb.org/anthology/N18-3005/
        [2] Data Collection for a Production Dialogue System: A Clinc Perspective
    """
    assert isinstance(s1, str) and isinstance(s2, str)
    assert isinstance(n, int) and n > 0
    if s1 == s2:
        return 0.0
    prob_sum = 0.0
    while n > 0:
        set_1 = set([s1[i: i + n] for i in range(len(s1) - n + 1)])
        set_2 = set([s2[i: i + n] for i in range(len(s2) - n + 1)])
        inter_sets = set_1.intersection(set_2)
        union_sets = set_1.union(set_2)
        prob_sum += safe_division(len(inter_sets), len(union_sets))
        n -= 1

    return 1.0 - prob_sum


def dataset_diversity(dataset: List[str], n: int = 3) -> Union[float, int]:
    r"""
    The diversity of a dataset is the average distance between all sentence pairs. It is slightly different
    with the original equation, because we do not consider senetence categories.

        DIV(D) = \frac{1}{|D|^2} [\sum_{a}^{D}\sum_{b}^{D} DIST(a, b)]

    Args:
        dataset (List[str]): A list of sentences.
        n (int): The maximum n-gram length.

    Reference:
        [1] https://www.aclweb.org/anthology/N18-3005/
        [2] Data Collection for a Production Dialogue System: A Clinc Perspective
    """
    if len(dataset) == 0:
        return 0.0

    dist_sum = 0.0
    for i, a in enumerate(dataset):
        for j, b in enumerate(dataset):
            dist_sum += sentence_distance(a, b, n)
    diversity = (1.0/(len(dataset)**2)) * dist_sum

    return diversity
