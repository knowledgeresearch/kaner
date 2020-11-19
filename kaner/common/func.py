# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Function utils"""

__all__ = ["query_time", "timing", "feed_args"]

import time
import inspect
from typing import Callable, Dict, Any


# Record the calculation duration for each function.
_TIME_RECORDS = {}


def timing(func):
    """
    Calculate the duration of running a given function and save it.

    Args:
        func (Callable): Function to be counted.
    """
    key = "{0}.{1}".format(func.__module__, func.__name__)

    def wrap(*args, **kwargs):
        time1 = time.time()
        returns = func(*args, **kwargs)
        time2 = time.time()
        duration = time2 - time1
        print('[Timing] {:s} function took {:.3f} sec'.format(key, duration))
        if key not in _TIME_RECORDS:
            _TIME_RECORDS[key] = []
        _TIME_RECORDS[key].append(duration)

        return returns
    wrap.__name__ = func.__name__
    wrap.__module__ = func.__module__
    return wrap


def query_time(func):
    """
    Query the latest duration of running a given function.

    Args:
        func (Callable): Function which has been counted.
    """
    key = "{0}.{1}".format(func.__module__, func.__name__)
    if key not in _TIME_RECORDS:
        return -1
    return _TIME_RECORDS[key][-1]


def feed_args(func: Callable, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find all avaiable arguments from options for a given function.

    Args:
        func (Callable): A given function to be feed arguments.
        options (Dict[str, Any]): Avaiable arguments.

    Examples:
        def func(a: int, b: str):
            print(a, b)
        options = {"a": 1, "b": "example", "c": 2}
        kwargs = feed_args(func, options)  # kwargs = {"a": 1, "b": "example"}
        func(**kwargs)  # 1, example
    """
    args = inspect.getfullargspec(func).args
    kwargs = {}
    for arg in args:
        if arg in options.keys():
            kwargs[arg] = options[arg]

    return kwargs
