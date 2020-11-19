# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Document utils"""

from typing import List

__all__ = ["slide_window"]


def slide_window(document: str, max_seq_len: int, back_offset: int) -> List[dict]:
    """
    Cut long document into small pieces.

    Args:
        text (str): The long document that needs to be cut..
        max_seq_len (int): .
        back_offset (int): .
    """
    texts = []
    pointer, segid = 0, 0
    while pointer < len(document):
        new_text = document[pointer: pointer + max_seq_len]
        texts.append(
            {
                "id": "DocPart-{0}".format(segid),
                "text": new_text,
                "start": pointer,
                "end": min(pointer + max_seq_len, len(document))
            }
        )
        segid += 1
        pointer += max_seq_len - back_offset

    return texts
