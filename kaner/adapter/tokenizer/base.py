# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Base Tokenizer"""

import os
from typing import Union, Dict, Any

import torch

from kaner.common import save_text, load_json, save_json


class BaseTokenizer:
    """
    BaseTokenizer defines the basic interfaces for the text pre-processing.

    Args:
        token_folder (str): The root folder of tokens.
        unk_token (str): Unknown token, such as '[UNK]'.
    """

    def __init__(self, token_folder: str, unk_token: str, pad_token: str):
        super(BaseTokenizer, self).__init__()
        self._token_folder = token_folder
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._to_ids = {}
        self._to_tokens = []
        self._load()
        assert self._unk_token in self._to_ids.keys()
        assert self._pad_token in self._to_ids.keys()

    def exist(self, token: str) -> bool:
        """
        Check whether the token exists in the tokenizer.

        Args:
            token (str): The token to be checked.
        """
        return token in self._to_ids.keys()

    def _load(self):
        """
        Load tokens.
        """
        token_path = os.path.join(self._token_folder, "tokens.txt")
        with open(token_path, "r") as f_in:
            for line in f_in.readlines():
                line = line.replace("\n", "")
                self._to_ids[line] = len(self._to_tokens)
                self._to_tokens.append(line)

    def save(self, folder: str):
        """
        Save tokens to a folder.

        Args:
            folder (str): The folder where tokens will save into.
        """
        save_text("\n".join(self._to_tokens), folder, "tokens.txt")
        save_json(self.configs(), folder, "token_configs.json")

    @property
    def unk_token(self):
        """
        Unknown token, such as '[UNK]'.
        """
        return self._unk_token

    @property
    def pad_id(self):
        """
        ID of PAD token, such as '<pad>'.
        """
        return self[self._pad_token]

    @property
    def pad_token(self):
        """
        PAD token, such as '<pad>'.
        """
        return self._pad_token

    @property
    def unk_id(self):
        """
        ID of Unknown token, such as '[UNK]'.
        """
        return self[self._unk_token]

    def __len__(self):
        """
        Return the number of tokens.
        """
        return len(self._to_tokens)

    def __getitem__(self, item):
        """
        Return the corresponding map value: id->token, token->id.
        """
        assert isinstance(item, (int, str))
        if isinstance(item, int):
            if 0 <= item < len(self):
                return self._to_tokens[item]
            return self.unk_token
        if item in self._to_ids.keys():
            return self._to_ids[item]
        return self.unk_id

    def tokenize(self, text: str):
        """
        Tokenize a text.

        Args:
            text (str): The text that will be tokenized.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: list):
        """
        Convert a list of token to the corresponding ids.

        Args:
            tokens (list): A list of token to be converted.
        """
        return [self[token] for token in tokens]

    def embeddings(self) -> Union[None, torch.FloatTensor]:
        """
        Return the pretrained token embeddings.
        """
        file_path = os.path.join(self._token_folder, "token_embeddings.checkpoints")
        embeddings = None
        if os.path.isfile(file_path):
            embeddings = torch.load(file_path)
        if embeddings is None:
            print("Token embedding is None!")

        return embeddings

    def configs(self) -> Dict[str, Any]:
        """
        Return the tokenizer configurations.
        """
        config = load_json("utf-8", self._token_folder, "token_configs.json")
        config["pad_id"] = self.pad_id

        return config
