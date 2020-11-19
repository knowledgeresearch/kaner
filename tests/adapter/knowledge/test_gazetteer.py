# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Gazetteer tests"""

import tempfile
import os
import torch
from kaner.common import save_text, save_json
from kaner.adapter.knowledge.gazetteer import Gazetteer


def test_gazetteer():
    """Test the class `Gazetteer`."""
    lexicons = [
        ("[PAD]", "SEP", "TEST"),
        ("南京", "LOC", "TEST"),
        ("南京市", "LOC", "TEST"),
        ("长江", "VIEW", "TEST"),
        ("长江大桥", "BUILDING", "TEST"),
        ("江大桥", "PER", "TEST"),
        ("大桥", "SEGMENTATION", "TEST")
    ]
    lexicon_embeddings = [
        [0.0, 0.0],
        [1.0, 0.1],
        [0.9, 0.3],
        [0.7, 0.8],
        [0.21, 0.78],
        [0.51, 0.82],
        [0.23, 0.91]
    ]
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(["\t".join(lex) for lex in lexicons]), folder_name, "lexicons.txt")
    torch.save(torch.tensor(lexicon_embeddings), os.path.join(folder_name, "lexicon_embeddings.checkpoints"))
    save_json({"n_lexicons": 7, "lexicon_dim": 2}, folder_name, "lexicon_configs.json")
    gazetteer = Gazetteer(folder_name)
    assert len(gazetteer) == 7
    assert gazetteer.configs() == {"n_lexicons": 7, "lexicon_dim": 2, "n_edge_types": 7}
    assert gazetteer.pad_token == "[PAD]"
    assert gazetteer.num_types == 6
    assert gazetteer["长江"] == 3
    assert gazetteer[4] == "长江大桥"
    assert gazetteer.search(["长", "江", "大", "桥"]) == ["长江", "长江大桥"]
    assert gazetteer.exist(["长", "江", "大", "桥"]) is True
    assert gazetteer.exist(["长", "江", "大"]) is False
    assert [[round(e, 2) for e in em] for em in gazetteer.embeddings().tolist()] == lexicon_embeddings
    assert gazetteer.freq("南京") == 0
    gazetteer.count_freq([{"text": "南京市长江大桥"}])
    assert gazetteer.freq("南京") == 1
    # update lexicons
    gazetteer.update(["{0}\tTEST\tTEST".format(lexicon) for lexicon in ["[PAD]", "重庆", "长江"]])
    assert len(gazetteer) == 3
    assert gazetteer.configs() == {"n_lexicons": 3, "lexicon_dim": 2, "n_edge_types": 2}
    assert gazetteer.pad_token == "[PAD]"
    assert gazetteer.num_types == 1
    assert gazetteer["长江"] == 2
    assert gazetteer["重庆市"] == 0
    assert gazetteer.embeddings() is None
    tmp_folder.cleanup()
    # test mask
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(["\t".join(lex) for lex in lexicons]), folder_name, "lexicons.txt")
    sentence = "南京市长江大桥"
    gazetteer = Gazetteer(folder_name)
    tmp_folder.cleanup()
    gazetteer.mask(["南京市", "长江大桥"], True)
    assert gazetteer.search(list(sentence)) == ["南京"]
    assert gazetteer.search(list(sentence[3:])) == ["长江"]
    gazetteer.mask(["南京市", "长江大桥"], False)
    assert gazetteer.search(list(sentence)) == ["南京", "南京市"]
    assert gazetteer.search(list(sentence[3:])) == ["长江", "长江大桥"]
