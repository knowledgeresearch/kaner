# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics: Named Entity Recognition"""

from typing import Dict, List, Any, Union
from collections import defaultdict

from kaner.metric.utils import safe_division
from kaner.adapter.knowledge import Gazetteer


def span_coverage_ratio(train_set: List[dict], test_set: List[dict]) -> Union[float, int]:
    r"""
    Calculate Span Coverage Ratio (SCR) of test set on the training set. All data's format should be as the same
    as the following sample:

        train/dev/test-set = [
            {"text": "abcdefg", "spans": [{"start": 0, "end": 1, "label": "example"}]}
        ]

    We use the following equation to calculate Span Coverage Ratio:

        p = \frac{\sum_{k=1}^{K} \frac{\#(e_i^{tr,k})}{C^{tr}} \#(e_i^{te,k})}{C^{te}}

    Args:
        train_set (List[dict]): Training set.
        test_set (List[dict]): Test set.

    References:
        [1] Rethinking Generalization of Neural Models: A Named Entity Recognition Case Study
        [2] https://github.com/pfliu-nlp/Named-Entity-Recognition-NER-Papers
    """
    labels = set()
    train_count, test_count = defaultdict(int), defaultdict(int)
    for sample in train_set:
        for span in sample["spans"]:
            labels.add(span["label"])
            train_count[span["label"]] += 1
    for sample in test_set:
        for span in sample["spans"]:
            labels.add(span["label"])
            test_count[span["label"]] += 1
    p = 0.0
    n_train_spans, n_test_spans = sum(train_count.values()), sum(test_count.values())
    for label in labels:
        p += safe_division(train_count[label], n_train_spans) * test_count[label]
    p = safe_division(p, n_test_spans)

    return p


def span_distribution(dataset: List[dict]) -> Dict[str, Any]:
    """
    Calculate sentence length distribution, average spans per sentence, label distribution, etc.
    Data's format should be as the same as the following sample:

        dataset = [
            {"text": "abcdefg", "spans": [{"start": 0, "end": 1, "label": "example"}]}
        ]
    Args:
        dataset (List[dict]): Training set.
    """
    results = {
        "sentence": {"max": 0, "min": 0, "avg": 0, "dist": defaultdict(int)},
        "span": {"max": 0, "min": 0, "avg": 0, "dist": defaultdict(int)},
        "label": {"dist": defaultdict(int)},
        "entities": set()
    }
    for sample in dataset:
        results["sentence"]["dist"][len(sample["text"])] += 1
        for ins in sample["spans"]:
            results["span"]["dist"][ins["end"] - ins["start"] + 1] += 1
            results["label"]["dist"][ins["label"]] += 1
            results["entities"].add(ins["text"])
    for category in ["sentence", "span", "label"]:
        if category != "label":
            results[category]["avg"] = safe_division(
                sum([length*count for length, count in results[category]["dist"].items()]), sum(results[category]["dist"].values())
            )
            lengths = list(results[category]["dist"].keys())
            if len(lengths) == 0:
                lengths += [0]
            results[category]["max"] = max(lengths)
            results[category]["min"] = min(lengths)
            results[category]["dist"] = sorted(list(results[category]["dist"].items()), key=lambda t: t[0])
        else:
            results[category]["dist"] = list(results[category]["dist"].items())
    results["entities"] = sorted(list(results["entities"]))

    return results


def lexicon_distribution(gazetteer: Gazetteer, dataset: List[dict]) -> Dict[str, Any]:
    """
    Given a dataset, compute matched lexicon distribution from a gazetteer.

    Args:
        gazetteer (Gazetteer): Gazetteer used to search matched lexicons.
        dataset (List[dict]): Dataset.
    """
    results = {
        "token": {"max": 0, "min": 0, "avg": 0, "dist": defaultdict(int)},
        "sentence": {"max": 0, "min": 0, "avg": 0, "dist": defaultdict(int)}
    }
    batch_tokens = [list(sample["text"]) for sample in dataset]
    batch_token_matched_lexicons = [[[] for _, _ in enumerate(sample["text"])] for _, sample in enumerate(dataset)]
    batch_sentence_matched_lenxicons = [[] for _, _ in enumerate(dataset)]
    for i, tokens in enumerate(batch_tokens):
        for j, _ in enumerate(tokens):
            matched_lexicons = gazetteer.search(tokens[j:])
            for lexicon in matched_lexicons:
                batch_sentence_matched_lenxicons[i].append(lexicon)
                for k in range(j, j + len(lexicon)):
                    batch_token_matched_lexicons[i][k].append(lexicon)
    # sentence level
    for _, sentence_matched_lexicons in enumerate(batch_sentence_matched_lenxicons):
        results["sentence"]["dist"][len(sentence_matched_lexicons)] += 1
    # token level
    for batch_id, _ in enumerate(batch_token_matched_lexicons):
        for token_pos_id, _ in enumerate(batch_token_matched_lexicons[batch_id]):
            results["token"]["dist"][len(batch_token_matched_lexicons[batch_id][token_pos_id])] += 1

    for category in ["sentence", "token"]:
        results[category]["avg"] = safe_division(
            sum([number*count for number, count in results[category]["dist"].items()]), sum(results[category]["dist"].values())
        )
        numbers = list(results[category]["dist"].keys())
        if len(numbers) == 0:
            numbers += [0]
        results[category]["max"] = max(numbers)
        results[category]["min"] = min(numbers)
        results[category]["dist"] = sorted(list(results[category]["dist"].items()), key=lambda t: t[0])

    # entity coverage ratio in gazetteer (gECR)
    total_count, matched_count = 0, 0
    for sample in dataset:
        for span in sample["spans"]:
            total_count += 1
            if gazetteer.exist(list(span["text"])):
                matched_count += 1
    results["gECR"] = safe_division(matched_count, total_count)

    return results
