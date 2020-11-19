# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Embedding"""

import math
from collections import OrderedDict
from typing import Union
import torch
import torch.nn as nn
from .bert import BERTModel


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding with the following equations:

        pe(pos, 2*i) = sin(pos/10000^(2*i/encoding_dim))
        pe(pos, 2*i+1) = cos(pos/10000^(2*i/encoding_dim))

    Reference:
        [1] https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        [2] http://nlp.seas.harvard.edu/2018/04/03/attention.html#embeddings-and-softmax

    Args:
        max_seq_len (int): The maximum length of a sequence.
        encoding_dim (int): The dimension of positional encoding.
        dropout (float): Dropout for positional embedding.
    """

    def __init__(self, max_seq_len: int, encoding_dim: int, dropout: float):
        super(SinusoidalPositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position_embedding = torch.zeros(max_seq_len, encoding_dim)
        pos = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, encoding_dim, 2) * (math.log(10000.0) / encoding_dim)
        )
        position_embedding[:, 0::2] = torch.sin(pos * div_term)
        position_embedding[:, 1::2] = torch.cos(pos * div_term)
        position_embedding.requires_grad = False
        self.register_buffer("pe", position_embedding)

    def forward(self, inputs):
        """
        Compute position embedding.

        Args:
            inputs (torch.LongTensor): Inputs with shape (batch_size, seq_len).
        """
        outputs = self.dropout(self.pe[:inputs.shape[1]])

        return outputs


class SoftwordEmbedding(nn.Module):
    """
    The implementation of the paper "Simplify the Usage of Lexicon in Chinese NER". SoftwordEmbedding
    to encodes the matched words, obtained from the lexicon, into the representations of characters.

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        n_lexicons (int): The number of lexicons in a gazetteer.
        lexicon_dim (int): The dimension of lexicon embedding.
        token_embedding_type (str): Token embedding type. Allowed values are 'static' or 'dynamic'. Static means static embeddings,
            including word2vec, glove, etc. Dynamic means dynamic embeddings, including various pretrained language models (BERT).
        token_embeddings (torch.FloatTensor): The pretrained token embeddings, default is None.
        lexicons_embeddings (torch.FloatTensor): The pretrained lexicon embeddings, default is None.
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            max_seq_len: int,
            n_token_types: int,
            pad_id: int,
            layer_norm_eps: float,
            hidden_dropout: float,
            attn_dropout: float,
            n_heads: int,
            intermediate_size: int,
            n_hidden_layers: int,
            n_lexicons: int,
            lexicon_dim: int,
            token_embedding_type: str = "static",
            token_embeddings: Union[torch.FloatTensor, OrderedDict] = None,
            lexicons_embeddings: torch.FloatTensor = None
    ):
        super(SoftwordEmbedding, self).__init__()
        assert isinstance(token_embedding_type, str) and token_embedding_type in ["static", "dynamic"]
        self.token_embedding_type = token_embedding_type
        if token_embedding_type == "static":
            if token_embeddings is not None:
                self.token_embeddings = nn.Embedding.from_pretrained(token_embeddings)
            else:
                self.token_embeddings = nn.Embedding(n_tokens, token_dim)
        elif token_embedding_type == "dynamic":
            self.bert = BERTModel(
                n_tokens, token_dim, max_seq_len, n_token_types, pad_id, n_heads, token_dim,
                intermediate_size, n_hidden_layers, attn_dropout, layer_norm_eps, hidden_dropout
            )
            if token_embeddings is not None:
                self.bert.load_state_dict(token_embeddings)
        if lexicons_embeddings is not None:
            self.lexicons_embeddings = nn.Embedding.from_pretrained(lexicons_embeddings)
        else:
            self.lexicons_embeddings = nn.Embedding(n_lexicons, lexicon_dim)

    def forward(self, inputs, lexicons, weights, position_ids=None, token_type_ids=None):
        """
        Combine character embedding with its corresponding lexicon embeddings.

        Args:
            inputs (torch.LongTensor): The character sequence with shape (batch_size, seq_len).
            lexicons (torch.LongTensor): Gazetteer set for each character in a character sequence.
                Expected shape is (batch_size, seq_len, n_sets , max_n_lexicons).
            weights (torch.FloatTensor): Weights for each character. Expected shape is the same as lexicons.
            position_ids (torch.LongTensor): Position ids with shape (batch_size, seq_len).
            token_type_ids (torch.LongTensor): Token type ids with shape (batch_size, seq_len).
        """
        batch_size, seq_len = inputs.shape
        # (batch_size, seq_len, token_dim)
        if self.token_embedding_type == "static":
            token_embeds = self.token_embeddings(inputs)
        else:
            _, token_embeds = self.bert(inputs, position_ids=position_ids, token_type_ids=token_type_ids)
        # (batch_size, seq_len, n_sets, max_n_lexicons, lexicon_dim)
        lexicon_embeds = self.lexicons_embeddings(lexicons)
        lexicon_embeds = torch.mul(
            lexicon_embeds,
            weights.unsqueeze(4).repeat(1, 1, 1, 1, lexicon_embeds.shape[-1])
        )
        # (batch_size, seq_len, n_sets*lexicon_dim)
        lexicon_embeds = lexicon_embeds.sum(dim=3).view(batch_size, seq_len, -1)
        # (batch_size, seq_len, token_dim + n_sets*lexicon_dim)
        outputs = torch.cat([token_embeds, lexicon_embeds], dim=-1)

        return outputs
