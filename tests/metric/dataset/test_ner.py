# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metric: Dataset Tests"""

import tempfile

from kaner.adapter.knowledge import Gazetteer
from kaner.common import save_text
from kaner.metric.dataset import (
    span_coverage_ratio,
    span_distribution,
    lexicon_distribution
)


def test_span_coverage_ratio():
    """Test the function `span_coverage_ratio`."""
    train_set = [
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "ORG"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "ORG"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "ORG"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "ORG"}]}
    ]
    test_set = [
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "PER"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "ORG"}]},
        {"text": "example", "spans": [{"start": 0, "end": 1, "label": "ORG"}]},
    ]
    assert round(span_coverage_ratio(train_set, test_set), 8) == 0.52
    assert span_coverage_ratio([], []) == 0.0


def test_span_distribution():
    """Test the function `span_distribution`."""
    results = {
        "sentence": {"dist": [], "avg": 0, "max": 0, "min": 0},
        "span": {"dist": [], "avg": 0, "max": 0, "min": 0},
        "label": {"dist": []},
        "entities": []
    }
    assert span_distribution([]) == results
    dataset = [
        {"text": "example", "spans": [{"start": 0, "end": 6, "label": "WORD", "text": "example"}]},
        {"text": "we", "spans": [{"start": 0, "end": 1, "label": "PER", "text": "we"}]}
    ]
    results = {
        "entities": ["example", "we"],
        "label": {"dist": [("WORD", 1), ("PER", 1)]},
        "sentence": {"dist": [(2, 1), (7, 1)], "avg": 4.5, "max": 7, "min": 2},
        "span": {"dist": [(2, 1), (7, 1)], "avg": 4.5, "max": 7, "min": 2}
    }
    assert span_distribution(dataset) == results


def test_lexicon_distribution():
    """Test the function `lexicon_distribution`."""
    lexicons = [
        ("[PAD]", "SEP", "TEST"),
        ("南京", "LOC", "TEST"),
        ("南京市", "LOC", "TEST"),
        ("长江", "VIEW", "TEST"),
        ("长江大桥", "BUILDING", "TEST"),
        ("江大桥", "PER", "TEST"),
        ("大桥", "SEGMENTATION", "TEST")
    ]
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(["\t".join(lex) for lex in lexicons]), folder_name, "lexicons.txt")
    gazetteer = Gazetteer(folder_name)
    results = {
        "sentence": {
            "dist": [],
            "avg": 0, "max": 0, "min": 0
        },
        "token": {
            "dist": [],
            "avg": 0, "max": 0, "min": 0
        },
        "gECR": 0.
    }
    assert lexicon_distribution(gazetteer, []) == results
    dataset = [
        {"text": "南京市长江大桥", "spans": [{"text": "南京", "label": "ORG", "start": 0, "end": 1}]},
        {"text": "重庆长江大桥", "spans": [{"text": "重庆", "label": "ORG", "start": 0, "end": 1}]}
    ]
    results = {
        "sentence": {
            "dist": [(4, 1), (6, 1)],
            "avg": 5.0, "max": 6, "min": 4
        },
        "token": {
            "dist": [(0, 2), (1, 1), (2, 4), (3, 6)],
            "avg": 27/13, "max": 3, "min": 0
        },
        "gECR": 0.5
    }
    assert lexicon_distribution(gazetteer, dataset) == results
    tmp_folder.cleanup()
