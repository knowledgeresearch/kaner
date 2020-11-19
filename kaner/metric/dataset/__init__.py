# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics for evaluation on datasets.
    Size, Diversity, Distribution, and Coverage.
"""

__all__ = [
    "sentence_distance",
    "dataset_diversity",
    "span_coverage_ratio",
    "span_distribution",
    "lexicon_distribution"
]


from .common import (
    sentence_distance,
    dataset_diversity
)
from .ner import (
    span_coverage_ratio,
    span_distribution,
    lexicon_distribution
)
