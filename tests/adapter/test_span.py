# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Span tests"""

from kaner.adapter.span import (
    Span,
    to_spans,
    to_tags,
    eliminate_overlap,
    split_document
)


def test_span():
    """Test the module `span`."""
    # test class `Span`
    span1 = Span("Test1", 3, 10, "Example", 1.0)
    span2 = Span("Test1", 3, 10, "Example", 1.0)
    span3 = Span("Test2", 3, 10, "Example", 1.0)
    assert span1 == span2
    assert span1 != span3
    # test function `to_spans`
    spans1 = to_spans(["O", "B-Test", "I-Test", "O"], ["A", "B", "C", "D"], [0.5, 0.2, 0.6, 0.8])
    assert len(spans1) == 1 and Span("Test", 1, 2, "BC", 0.4) == spans1[0]
    spans2 = to_spans(["O", "B-Test", "I-Test"], ["A", "B", "C"], [0.3] * 3)
    assert len(spans2) == 1 and Span("Test", 1, 2, "BC", 0.3) == spans2[0]
    spans3 = to_spans(["O", "O", "O"], ["A", "B", "C"], [0.3] * 3)
    assert len(spans3) == 0
    spans4 = to_spans([], [], [])
    assert len(spans4) == 0
    spans5 = to_spans(["O", "B-Test", "I-Test", "B-Test2"], ["A", "B", "C", "D"], [0.9] * 4)
    assert len(spans5) == 2 and Span("Test", 1, 2, "BC", 0.9) == spans5[0] and Span("Test2", 3, 3, "D", 0.9) == spans5[1]
    # test function `to_tags`
    tags1 = to_tags(4, [Span("Test", 1, 2, "BC", 1.0), Span("Test2", 3, 3, "D", 1.0)])
    assert tags1 == ["O", "B-Test", "I-Test", "B-Test2"]
    tags1 = to_tags(4, [])
    assert tags1 == ["O", "O", "O", "O"]
    # test func `eliminate_overlap`
    intervals = [(2, 5), (1, 3), (1, 2), (2, 8), (9, 10), (2, 3), (1, 5), (2, 5)]
    spans = [Span("Test", interval[0], interval[1], "Example", 1.0) for interval in intervals]
    answers = [Span("Test", interval[0], interval[1], "Example", 1.0) for interval in [(1, 5), (9, 10)]]
    new_spans = eliminate_overlap(spans)
    assert "\t".join([str(span) for span in new_spans]) == "\t".join([str(span) for span in answers])


def test_split_document():
    """Test the function `split_document`."""
    document = "abc。ae。aeeeeeeee。uio。aa。"
    spans = [Span("example", 0, 1, "ab", 1.0), Span("example", 4, 5, "ae", 1.0), Span("example", 11, 13, "ee", 1.0)]
    true_data = [
        {"text": "abc。", "spans": [Span("example", 0, 1, "ab", 1.0)]},
        {"text": "ae。", "spans": [Span("example", 0, 1, "ae", 1.0)]},
        {"text": "uio。", "spans": []},
        {"text": "aa。", "spans": []}
    ]
    true_ignored = [
        {"text": "aeeeeeeee。", "spans": [Span("example", 4, 6, "ee", 1.0)]}
    ]
    data, ignored = split_document(document, spans, 5)
    assert data == true_data and ignored == true_ignored
    assert split_document("", [], 5) == ([], [])
    assert split_document("。", [], 5) == ([{"text": "。", "spans": []}], [])
    assert split_document("abcde", [], 5) == ([{"text": "abcde", "spans": []}], [])
