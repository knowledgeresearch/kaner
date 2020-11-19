# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Gazetteer"""

import os
from typing import List, Union, Dict, Any

import torch

from kaner.common import save_text, load_json, save_json
from .trie import TrieTree


class Gazetteer:
    """
    Gazetteer provides the map between entity and index, and quick searching based on trie tree.

    Args:
        gazetteer_folder (str): The root folder where the gazetteer exists.
    """

    def __init__(self, gazetteer_folder: str):
        super(Gazetteer, self).__init__()
        self._pad_token = "[PAD]"
        self._gazetteer_folder = gazetteer_folder
        self._to_ids = {}
        self._to_lexicons = []
        self._to_attributes = []  # additional attributes
        self._embeddings = None
        self._trie = TrieTree()
        self.update(None)

    def update(self, lexicons: List[str] = None) -> None:
        """
        Given a gazetteer folder or a list of lexicon, load lexicons into memory.

        Args:
            lexicons (List[str]): A list of lexicon.
        """
        self._to_ids.clear()
        self._to_lexicons.clear()
        self._to_attributes.clear()
        self._trie = TrieTree()
        if lexicons is None:
            gazetteer_path = os.path.join(self._gazetteer_folder, "lexicons.txt")
            with open(gazetteer_path, "r") as f_in:
                lexicons = [line.replace("\n", "") for line in f_in.readlines()]
            file_path = os.path.join(self._gazetteer_folder, "lexicon_embeddings.checkpoints")
            if os.path.isfile(file_path):
                self._embeddings = torch.load(file_path)
        else:
            self._embeddings = None
        if self._embeddings is None:
            print("Lexicon embedding is None! {0}".format(self._gazetteer_folder))
        for lexicon in lexicons:
            columns = lexicon.split("\t")
            assert len(columns) == 3, "The format of lexicon line must be lexicon\\ttype\\tsource."
            lexicon, lexicon_type, lexicon_source = columns
            self._trie.insert(list(lexicon))
            self._to_ids[lexicon] = len(self._to_lexicons)
            self._to_lexicons.append(lexicon)
            self._to_attributes.append({"lexicon_type": lexicon_type, "lexicon_source": lexicon_source, "lexicon_freq": 0})
        assert self._pad_token in self._to_ids.keys()

    def mask(self, item_or_items: Union[str, list], status: bool = True) -> None:
        """
        Mask (unmask) an item or a list of items. If an item is masked, then it will be blocked.

        item_or_items (Union[str, list]): An item or a list of items to masked (unmasked).
        status (bool): If the status is True, then execute mask; Otherwise, exexcute unmask.
        """
        if status:
            self._trie.mask(item_or_items)
        else:
            self._trie.clear_mask(item_or_items)

    def save(self, folder: str) -> None:
        """
        Save gazetteer to a folder.

        Args:
            folder (str): The folder where gazetteers will save into.
        """
        lexicons = []
        for lexicon in self._to_lexicons:
            lex_id = self._to_ids[lexicon]
            lexicon_type = self._to_attributes[lex_id]["lexicon_type"]
            lexicon_source = self._to_attributes[lex_id]["lexicon_source"]
            lexicons.append("{0}\t{1}\t{2}".format(lexicon, lexicon_type, lexicon_source))
        save_text("\n".join(lexicons), folder, "lexicons.txt")
        save_json(self.configs(), folder, "lexicon_configs.json")

    @property
    def pad_token(self) -> str:
        """
        PAD token.
        """
        return self._pad_token

    @property
    def pad_id(self) -> int:
        """
        ID of PAD token.
        """
        return self[self._pad_token]

    def __len__(self):
        """
        Return the number of lexicons.
        """
        return len(self._to_lexicons)

    def __getitem__(self, item):
        """
        Return the corresponding map value: id->lexicon, lexicon->id.
        """
        assert isinstance(item, (int, str))
        if isinstance(item, int):
            if 0 <= item < len(self):
                return self._to_lexicons[item]
            return self.pad_token
        if item in self._to_ids.keys():
            return self._to_ids[item]
        return self.pad_id

    def lexicon_type(self, item):
        """
        Return the corresponding lexicon type: lexicon_id->lexicon_type, lexicon_id->lexicon_type.
        """
        assert isinstance(item, (int, str))
        if isinstance(item, str):
            lex_id = self[item]
        elif 0 <= lex_id < len(self):
            lex_id = item
        else:
            lex_id = self.pad_id

        return self._to_attributes[lex_id]["lexicon_type"]

    @property
    def num_types(self):
        """
        Return the total number of lexicon types.
        """
        lexicon_types = set()
        for attrib in self._to_attributes:
            lexicon_types.add(attrib["lexicon_type"])

        return len(lexicon_types)

    def search(self, tokens: list):
        """
        Given a token list, search its all matched lexicons from the first token.

        Args:
            tokens (list): Token list.
        """
        return self._trie.enumerate_match(tokens)

    def exist(self, tokens: List[str]) -> bool:
        """
        Check whether a term is in this tree.

        Args:
            tokens (list): The component of this term. Usually, it is a list of character.
        """
        return self._trie.search(tokens)

    def embeddings(self) -> Union[None, torch.FloatTensor]:
        """
        Return the pretrained lexicon embeddings.
        """
        return self._embeddings

    def configs(self) -> Dict[str, Any]:
        """
        Return the lexicon configurations.
        """
        config = load_json("utf-8", self._gazetteer_folder, "lexicon_configs.json")
        config["n_lexicons"] = len(self)
        # character + n_lexicon_types
        config["n_edge_types"] = self.num_types + 1
        if "token_embedding_type" in config.keys():
            config.pop("token_embedding_type")

        return config

    def count_freq(self, dataset: List[dict]) -> None:
        """
        Count the lexicon frequence from the dataset.

        Args:
            dataset (List[dict]): Dataset list.
        """
        for sample in dataset:
            tokens = list(sample["text"])
            for i, _ in enumerate(tokens):
                matched_lexicons = self.search(tokens[i:])
                for lexicon in matched_lexicons:
                    lex_id = self[lexicon]
                    self._to_attributes[lex_id]["lexicon_freq"] += 1

    def freq(self, lexicon: str) -> int:
        """
        Get the frequency of a lexicon in a dataset.

        Args:
            lexicon (str): Lexicon to be queried.
        """
        lex_id = self[lexicon]

        return self._to_attributes[lex_id]["lexicon_freq"]
