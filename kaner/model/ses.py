# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""SES"""

from typing import Union
from collections import OrderedDict
import torch
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from .layers import CRF, SoftwordEmbedding


@gctx.register_model("ses")
class SES(nn.Module):
    """
    SES integrates gazetteer information by the improved softword technique.

    Args:
        hidden_dim (int): The dimension of hidden state.
        n_sets (int): The number of segment labels (BMES) which indicates the number of lexicon sets for each character.
        n_layers (int): The number of LSTM layers.
        n_tags (int): The number of tags for labeling sequence.

    References:
        [1] https://github.com/v-mipeng/LexiconAugmentedNER
        [2] https://arxiv.org/pdf/1908.05969.pdf
    """

    def __init__(
            self,
            n_tokens: int,
            n_tags: int,
            token_dim: int,
            n_lexicons: int,
            lexicon_dim: int,
            n_sets: int,
            n_layers: int,
            hidden_dim: int,
            pad_id: int,
            max_seq_len: int = None,
            n_token_types: int = None,
            layer_norm_eps: float = None,
            hidden_dropout: float = None,
            attn_dropout: float = None,
            n_heads: int = None,
            intermediate_size: int = None,
            n_hidden_layers: int = None,
            token_embedding_type: str = "static",
            token_embeddings: Union[torch.FloatTensor, OrderedDict] = None,
            lexicon_embeddings: torch.FloatTensor = None
    ):
        super(SES, self).__init__()
        self.embedding = SoftwordEmbedding(
            n_tokens, token_dim, max_seq_len, n_token_types, pad_id, layer_norm_eps, hidden_dropout, attn_dropout, n_heads,
            intermediate_size, n_hidden_layers, n_lexicons, lexicon_dim, token_embedding_type, token_embeddings, lexicon_embeddings
        )
        self.lstm = nn.LSTM(
            token_dim + n_sets*lexicon_dim, hidden_dim // 2, n_layers, batch_first=True, bidirectional=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.crf = CRF(n_tags, -1, pad_id)

    def forward(self, inputs, lexicons, weights, mask, tags=None):
        """
        Compute predictions or loss if tags is not None.

            -logp(y|x) = exp(f(y,x)) / sum(exp(f(y',x)))

        Args:
            inputs (torch.LongTensor): The sequence input with expected shape (batch_size, seq_len).
            lexicons (torch.LongTensor): Lexicon sets for each character in a character sequence. Expected shape is
                (batch_size, seq_len, n_sets, max_n_lexicons).
            weights (torch.FloatTensor): Weights for each character. Expected shape is the same as lexicons.
            mask (torch.FloatTensor): Batch sequence length with shape (batch_size, seq_len).
            tags (torch.LongTensor): The ground-truth tags with expected shape (batch_size, seq_len).
        """
        embeds = self.embedding(inputs, lexicons, weights)
        self.lstm.flatten_parameters()
        hidden_states, _ = self.lstm(embeds)
        emissions = self.hidden2tag(hidden_states)
        preds_or_loss = self.crf(emissions, mask, tags, True)

        return preds_or_loss
