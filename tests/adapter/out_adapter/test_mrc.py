# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""OutMRC tests"""

from kaner.adapter.out_adapter.mrc import OutMRC


def test_outseqlab():
    """Test the class `OutMRC`."""
    out_adapter = OutMRC()
    assert len(out_adapter) == 3
    assert out_adapter.unk_id == 0
    assert out_adapter.unk_label == "O"
    assert out_adapter.convert_labels_to_ids(["O", "O", "B-Span"]) == [0, 0, 1]
