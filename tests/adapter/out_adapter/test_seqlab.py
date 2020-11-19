# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""OutSeqlab tests"""

import tempfile
import os

from kaner.common import save_text
from kaner.adapter.out_adapter.seqlab import OutSeqlab


def test_outseqlab():
    """Test the class `OutSeqlab`."""
    labels = ["O", "B-Example", "I-Example"]
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(labels), folder_name, "labels")
    out_adapter = OutSeqlab(folder_name, "labels")
    assert len(out_adapter) == 3
    assert out_adapter.unk_id == 0
    assert out_adapter.unk_label == "O"
    assert out_adapter.convert_labels_to_ids(["O", "O", "B-Example"]) == [0, 0, 1]
    out_adapter.save(folder_name, "test")
    with open(os.path.join(folder_name, "test"), "r") as f_in:
        text = f_in.read()
    assert text == "\n".join(labels)
    tmp_folder.cleanup()
