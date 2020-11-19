# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Batcher Tests"""

from kaner.adapter.batcher import (
    batch_seq_mask,
    cfn_blcrf,
    cfn_ses,
    cfn_cgn,
    cfn_mdgg
)


def test_batch_seq_mask():
    """
    Test the function `batch_seq_mask`.
    """
    mask = batch_seq_mask([2, 5, 3])
    batch_mask = [
        [False, False, True, True, True],
        [False, False, False, False, False],
        [False, False, False, True, True]
    ]
    assert mask == batch_mask


def test_cfn_blcrf():
    """"
    Test the function `cfn_blcrf`
    """
    collate_fn = cfn_blcrf(0, 0, "cpu")
    batch_data = (
        {"input_ids": [1, 5, 2], "output_ids": [2, 1, 0], "length": 3},
        {"input_ids": [2, 2, 10, 9], "output_ids": [1, 1, 2, 2], "length": 4}
    )
    _, ready_to_eat = collate_fn(batch_data)
    assert str(ready_to_eat["inputs"].device) == "cpu" and str(ready_to_eat["outputs"].device) == "cpu"
    assert ready_to_eat["inputs"].tolist() == [[1, 5, 2, 0], [2, 2, 10, 9]]
    assert ready_to_eat["outputs"].tolist() == [[2, 1, 0, 0], [1, 1, 2, 2]]
    assert ready_to_eat["lengths"].tolist() == [3, 4]


def test_cfn_ses():
    """"
    Test the function `cfn_ses`
    """
    collate_fn = cfn_ses(0, 0, 100, "cpu")
    batch_data = (
        {
            "input_ids": [1], "output_ids": [2], "length": 1,
            "lexicon_ids": [[[1, 2], [2], [2], [1]]],
            "weights": [[[0.5, 0.5], [1.0], [1.0], [1.0]]]
        },
        {
            "input_ids": [2, 1], "output_ids": [1, 1], "length": 2,
            "lexicon_ids": [[[1], [2], [10], []], [[1], [], [], []]],
            "weights": [[[1.0], [1.0], [1.0], []], [[1.0], [], [], []]]
        }
    )
    _, ready_to_eat = collate_fn(batch_data)
    assert str(ready_to_eat["inputs"].device) == "cpu" and str(ready_to_eat["outputs"].device) == "cpu"
    assert ready_to_eat["inputs"].tolist() == [[1, 0], [2, 1]]
    assert ready_to_eat["outputs"].tolist() == [[2, 0], [1, 1]]
    assert ready_to_eat["lengths"].tolist() == [1, 2]
    # (batch_size, seq_len, n_sets, n_lexicons)
    batch_lexicons = [
        [[[1, 2], [2, 100], [2, 100], [1, 100]], [[100, 100], [100, 100], [100, 100], [100, 100]]],
        [[[1, 100], [2, 100], [10, 100], [100, 100]], [[1, 100], [100, 100], [100, 100], [100, 100]]]
    ]
    batch_weights = [
        [[[0.5, 0.5], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    ]
    assert ready_to_eat["lexicons"].tolist() == batch_lexicons
    assert ready_to_eat["weights"].tolist() == batch_weights


def test_cfn_cgn():
    """"
    Test the function `cfn_cgn`
    """
    collate_fn = cfn_cgn(0, 0, 100, "cpu")
    batch_data = (
        {
            "input_ids": [1, 5], "output_ids": [2, 1], "length": 2,
            "lexicon_ids": [2, 10],
            "relations": [[((0, True), (0, False))], [], []]
        },
        {
            "input_ids": [2], "output_ids": [1], "length": 1,
            "lexicon_ids": [8],
            "relations": [[((0, True), (0, True))], [], []]
        }
    )
    _, ready_to_eat = collate_fn(batch_data)
    batch_graphs = [
        [
            [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ],
        [
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ]
    ]
    assert str(ready_to_eat["inputs"].device) == "cpu" and str(ready_to_eat["outputs"].device) == "cpu"
    assert ready_to_eat["inputs"].tolist() == [[1, 5], [2, 0]]
    assert ready_to_eat["outputs"].tolist() == [[2, 1], [1, 0]]
    assert ready_to_eat["lengths"].tolist() == [2, 1]
    assert ready_to_eat["lexicons"].tolist() == [[2, 10], [8, 100]]
    assert ready_to_eat["graphs"].tolist() == batch_graphs


def test_cfn_mdgg():
    """"
    Test the function `cfn_mdgg`
    """
    collate_fn = cfn_mdgg(0, 0, 100, "cpu")
    batch_data = (
        {
            "input_ids": [1, 5], "output_ids": [2, 1], "length": 2,
            "lexicon_ids": [2, 10],
            "relations": [[((0, True), (0, False))]]
        },
        {
            "input_ids": [2], "output_ids": [1], "length": 1,
            "lexicon_ids": [8],
            "relations": [[((0, True), (0, True))]]
        }
    )
    _, ready_to_eat = collate_fn(batch_data)
    batch_graphs = [
        [[[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]]],
        [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]
    ]
    assert str(ready_to_eat["inputs"].device) == "cpu" and str(ready_to_eat["outputs"].device) == "cpu"
    assert ready_to_eat["inputs"].tolist() == [[1, 5], [2, 0]]
    assert ready_to_eat["outputs"].tolist() == [[2, 1], [1, 0]]
    assert ready_to_eat["lengths"].tolist() == [2, 1]
    assert ready_to_eat["lexicons"].tolist() == [[2, 10], [8, 100]]
    assert ready_to_eat["graphs"].tolist() == batch_graphs
