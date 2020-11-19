# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Character Tokenizer"""


from .base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    CharTokenizer that regrads one character as one token.

    Args:
        token_folder (str): The root folder of tokens.
    """

    def __init__(self, token_folder: str):
        super(CharTokenizer, self).__init__(token_folder, "[UNK]", "[PAD]")

    def tokenize(self, text: str):
        """
        Tokenize a text.

        Args:
            text (str): The text that will be tokenized.
        """
        return list(text)
