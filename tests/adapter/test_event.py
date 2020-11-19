# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Event tests"""

from kaner.adapter.event import Event
from kaner.adapter.span import Span


def test_event():
    """Test the module `event`."""
    arguments = [Span("公司名称", 0, 1, "九芝堂股份有限公司", 1.0), Span("公告时间", 2, 3, "2008年12月30日", 1.0)]
    event = Event("重大资产损失", 1.0, arguments)
    assert vars(event) == {
        'event_type': '重大资产损失',
        'confidence': 1.0,
        'arguments': [
            {'start': 0, 'end': 1, 'label': '公司名称', 'text': '九芝堂股份有限公司', 'confidence': 1.0},
            {'start': 2, 'end': 3, 'label': '公告时间', 'text': '2008年12月30日', 'confidence': 1.0}
        ]
    }
