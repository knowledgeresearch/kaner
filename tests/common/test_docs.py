# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Document utils tests"""

from kaner.common.docs import slide_window


def test_slide_window():
    assert slide_window("", 3, 1) == []
    assert slide_window("abcdefghigk", 3, 1) == [
        {"id": "DocPart-0", "text": "abc", "start": 0, "end": 3},
        {"id": "DocPart-1", "text": "cde", "start": 2, "end": 5},
        {"id": "DocPart-2", "text": "efg", "start": 4, "end": 7},
        {"id": "DocPart-3", "text": "ghi", "start": 6, "end": 9},
        {"id": "DocPart-4", "text": "igk", "start": 8, "end": 11},
        {"id": "DocPart-5", "text": "k", "start": 10, "end": 11}
    ]
