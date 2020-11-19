# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""CharTokenizer tests"""

import tempfile
import os
import torch
from kaner.common import save_text, save_json
from kaner.adapter.tokenizer.char import CharTokenizer


def test_chartokenizer():
    """Test the class `CharTokenizer`."""
    tokens = ["[UNK]", "[PAD]", "南", "京", "市", "长", "江", "大", "桥"]
    token_embeddings = [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.9, 0.3],
        [0.7, 0.8],
        [0.21, 0.78],
        [0.51, 0.82],
        [0.23, 0.91],
        [0.39, 0.61],
        [0.98, 0.45]
    ]
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    save_text("\n".join(tokens), folder_name, "tokens.txt")
    torch.save(torch.tensor(token_embeddings), os.path.join(folder_name, "token_embeddings.checkpoints"))
    config = {"n_tokens": 9, "token_dim": 2}
    save_json(config, folder_name, "token_configs.json")
    tokenizer = CharTokenizer(folder_name)
    assert len(tokenizer) == 9
    config["pad_id"] = 1
    assert tokenizer.configs() == config
    assert tokenizer.unk_token == "[UNK]"
    assert tokenizer.pad_token == "[PAD]"
    assert tokenizer["南"] == 2
    assert tokenizer[3] == "京"
    assert tokenizer.tokenize("南京是好朋友") == ["南", "京", "是", "好", "朋", "友"]
    assert tokenizer.convert_tokens_to_ids(["南", "京", "是", "好", "朋", "友"]) == [2, 3, 0, 0, 0, 0]
    assert [[round(e, 2) for e in em] for em in tokenizer.embeddings().tolist()] == token_embeddings
    tmp_folder.cleanup()
