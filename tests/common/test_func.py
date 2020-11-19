# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Funcition utils tests"""

import time
from kaner.common.func import (
    query_time,
    timing,
    feed_args
)


@timing
def _func_time_count():
    """
    This function is only used for counting time.
    """
    i, j, gate = 0, 0, 5000
    while i < gate:
        j = 0
        while j < gate:
            j += 1
        i += 1


def test_feed_args():
    """Test the function `feed_args`."""
    def feed_args_target(a: int, b: str) -> str:
        """The target function of `feed_args`."""
        return "{0}-{1}".format(a, b)

    options = {"a": 1, "b": "example", "c": 2}
    kwargs = feed_args(feed_args_target, options)
    assert feed_args_target(**kwargs) == "{0}-{1}".format(options["a"], options["b"])


def test_func():
    """Test the module `func`."""
    start = time.time()
    _func_time_count()
    end = time.time()
    eps = 1.0
    real_duration = end - start
    queried_duration = query_time(_func_time_count)
    assert abs(real_duration - queried_duration) < eps
