# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Input Adapter Tests on Seqlab"""

import tempfile

from kaner.adapter.in_adapter.seqlab import InSeqlab
from kaner.adapter.tokenizer.char import CharTokenizer
from kaner.adapter.out_adapter.mrc import OutMRC
from kaner.adapter.out_adapter.seqlab import OutSeqlab
from kaner.common import save_text


def test_inseqlab():
    """Test the class `InSeqlab`."""
    # test normal mode
    max_seq_len = 8
    dataset = [
        {
            "text": "abcdefghijk",
            "spans": [
                {"label": "Example", "start": 1, "end": 2, "text": "ex", "confidence": 1.0}
            ]
        },
        {
            "text": "a",
            "spans": []
        }
    ]
    tokens = ["[UNK]", "[PAD]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    labels = ["O", "B-Example", "I-Example"]
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(tokens), folder_name, "tokens.txt")
    save_text("\n".join(labels), folder_name, "labels.txt")
    tokenizer = CharTokenizer(folder_name)
    out_adapter = OutSeqlab(folder_name, "labels.txt")
    in_adapter = InSeqlab(dataset, max_seq_len, tokenizer, out_adapter)
    assert in_adapter.transform_sample(dataset[1]) == \
        {"text": "a", "input_ids": [2], "output_ids": [0], "length": 1, "start": 0}
    assert in_adapter.transform_sample(dataset[0]) == \
        {"text": "abcdefghijk", "input_ids": [2, 3, 4, 5, 6, 7, 8, 9], "output_ids": [0, 1, 2, 0, 0, 0, 0, 0], "length": 8, "start": 0}
    assert len(in_adapter) == 2
    tmp_folder.cleanup()

    # test MRC mode
    max_seq_len = 8
    dataset = [
        {
            "text": "acbq",
            "spans": [
                {"label": "T1", "start": 1, "end": 1, "text": "c", "confidence": 1.0},
                {"label": "T2", "start": 2, "end": 3, "text": "bq", "confidence": 1.0}
            ]
        }
    ]
    tokens = ["[UNK]", "[PAD]", "a", "b", "c", "q", "1", "2", "[CLS]", "[SEP]"]
    queries = {"T1": "q1", "T2": "q2"}
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(tokens), folder_name, "tokens.txt")
    tokenizer = CharTokenizer(folder_name)
    out_adapter = OutMRC()
    in_adapter = InSeqlab(dataset, max_seq_len, tokenizer, out_adapter, queries=queries)
    assert in_adapter.transform_sample(dataset[0]) == [
        {
            "type": "T1", "query": "q1", "text": "acbq",
            "input_ids": [8, 5, 6, 9, 2, 4, 3, 5], "output_ids": [0, 0, 0, 0, 0, 1, 0, 0], "length": 8, "start": 4
        },
        {
            "type": "T2", "query": "q2", "text": "acbq",
            "input_ids": [8, 5, 7, 9, 2, 4, 3, 5], "output_ids": [0, 0, 0, 0, 0, 0, 1, 2], "length": 8, "start": 4
        }
    ]
    assert len(in_adapter) == 2
    tmp_folder.cleanup()
