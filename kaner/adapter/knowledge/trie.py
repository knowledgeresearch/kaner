# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Trie Tree"""

__all__ = ["TrieTree"]

import os
from collections import defaultdict
from typing import List, Union


class _TrieNode:
    """
    Define the node of a Trie tree.

    Args:
        token (str): The token saved in the current node.
    """

    def __init__(self, token: str):
        super(_TrieNode, self).__init__()
        self.token = token
        self.children = defaultdict(_TrieNode)
        self.is_term = False


class TrieTree:
    """
    In computer science, a trie, also called digital tree or prefix tree, is a kind of search tree—an ordered tree data structure
    used to store a dynamic set or associative array where the keys are usually strings.

    Reference:
        [1] https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        super(TrieTree, self).__init__()
        self._root = _TrieNode("")
        self._len = 0
        self._masked_items = set()

    def mask(self, item_or_items: Union[str, list]) -> None:
        """
        Mask an item or a list of items, so it can not be searched.

        Args:
            items (Union[str, list]): A masked item or a list of masked items to be masked.
        """
        if isinstance(item_or_items, str):
            self._masked_items.add(item_or_items)
        elif isinstance(item_or_items, list):
            for item in item_or_items:
                assert isinstance(item, str)
                self._masked_items.add(item)

    def clear_mask(self, item_or_items: Union[str, list]) -> None:
        """
        Recover a masked item or a list of maksed items, so it can be searched again.

        Args:
            items (Union[str, list]): A masked item or a list of masked items to be masked.
        """
        if isinstance(item_or_items, str) and item_or_items in self._masked_items:
            self._masked_items.remove(item_or_items)
        elif isinstance(item_or_items, list):
            for item in item_or_items:
                if item in self._masked_items:
                    self._masked_items.remove(item)

    def insert(self, tokens: List[str]):
        """
        Insert a term into the tree. A term can be a word or phrase.

        Args:
            tokens (List[str]): The component of this term. Usually, it is a list of character.
        """
        cur = self._root
        for token in tokens:
            if token not in cur.children:
                cur.children[token] = _TrieNode(token)
            cur = cur.children[token]
        cur.is_term = True
        self._len += 1

    def search(self, tokens: List[str]) -> bool:
        """
        Search a term int the tree. A term can be a word or phrase.

        Args:
            tokens (List[str]): The component of this term. Usually, it is a list of character.
        """
        item = "".join(tokens)
        if item in self._masked_items:
            return False

        cur = self._root
        for token in tokens:
            if token not in cur.children:
                return False
            cur = cur.children[token]

        return cur.is_term

    def enumerate_match(self, prefix: List[str]) -> List[str]:
        """
        Enumerate all matched terms according to the prefix.

        Args:
            prefix (List[str]): A list of prefix token.
        """
        matched_terms = []
        cur = self._root
        for i, token in enumerate(prefix):
            if token not in cur.children:
                break
            cur = cur.children[token]
            if cur.is_term:
                item = "".join(prefix[:i+1])
                if item in self._masked_items:
                    continue
                else:
                    matched_terms.append(item)

        return matched_terms

    def load_lexicons(self, folder: str, file_name: str):
        """
        Load all lexicons into the trie tree.

        Args:
            folder (str): The root folder of the gazetteer.
            file_name (str): The file name of the gazetteer.
        """
        file_path = os.path.join(folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f_in:
            for line in f_in.readlines():
                line = line.replace("\n", "")
                tokens = list(line)
                self.insert(tokens)

    def __len__(self):
        """
        Return the total number of terms.
        """
        return self._len
