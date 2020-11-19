# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Machine Reading Comprehension Utils"""

import os
from typing import Dict
from kaner.common import save_text

__all__ = ["load_query", "save_query"]


_QUERY_TEMPLATE = [
    "type,question",
    "type1,question1",
    "type2,question2"
]
_QUERY_EXAMPLE = [
    "type,question",
    "破产清算-受理法院,找到破产清算事件的受理法院，包含各级解决社会矛盾的人民法院。"
    "破产清算-公告时间,找到破产清算事件的公告时间，包含年、月、日、天、周、时、分、秒等。",
    "重大安全事故-公告时间,找到重大安全事故事件的公告时间，包含年、月、日、天、周、时、分、秒等。",
    "重大安全事故-损失金额,找到重大安全事故事件的损失金额，包含分、角、元、万、亿等费用单位或数量。"
]


def _warning_info(file_path: str) -> None:
    """
    If the query file does not conform the standard formath, then print warnings.

    Args:
        file_path (str): The path of query file.
    """
    print("\033[91m# The query file {0} does not satisfy the need of MRC mode.\033[0m".format(file_path))
    print(" ", "1. Template")
    print("       " + "\n      ".join(_QUERY_TEMPLATE))
    print(" ", "2. Example")
    print("       " + "\n       ".join(_QUERY_EXAMPLE))
    exit(0)


def load_query(*paths) -> Dict[str, str]:
    """
    Load all queries from the local disk in the MRC mode.

    Args:
        paths (tuple): The query path tuple.
    """
    file_path = os.path.join(*paths)
    if not os.path.isfile(file_path):
        _warning_info(file_path)
    queries = {}
    with open(file_path, "r", encoding="utf-8") as f_in:
        for i, line in enumerate(f_in.readlines()):
            line = line.replace("\n", "")
            if line == "":
                continue
            if i == 0:
                if line != _QUERY_TEMPLATE[0]:
                    _warning_info(file_path)
            else:
                row = line.split(",")
                if len(row) != 2:
                    _warning_info(file_path)
                entity_type, query = row
                queries[entity_type] = query
    if len(queries) == 0:
        _warning_info(file_path)

    return queries


def save_query(queries: Dict[str, str], *paths) -> None:
    """
    Save all queries into the local disk in the MRC mode.

    Args:
        queries (Dict[str, str]): all queries for all span types.
        paths (tuple): The query path tuple.
    """
    rows = [_QUERY_TEMPLATE[0]] + ["{0},{1}".format(k, v) for k, v in queries.items()]
    save_text("\n".join(rows), *paths)
