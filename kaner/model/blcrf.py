# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""BiLSTMCRF"""

import torch
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from .layers import CRF


@gctx.register_model("blcrf")
class BLCRF(nn.Module):
    """
    A BiLSTMCRF model for sequence labeling.

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        n_tags (int): The number of tags.
        n_layers (int): The number of LSTM layers.
        hidden_dim (int): The dimension of hidden state.
        token_embeddings (torch.FloatTensor): The pretrained token embeddings, default is None.

    References:
        [1] https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
            #bi-lstm-conditional-random-field-discussion
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            n_tags: int,
            n_layers: int,
            hidden_dim: int,
            pad_id: int,
            token_embeddings: torch.Tensor = None
    ):
        super(BLCRF, self).__init__()
        if token_embeddings is not None:
            self.token_embeddings = nn.Embedding.from_pretrained(token_embeddings)
        else:
            self.token_embeddings = nn.Embedding(n_tokens, token_dim)
        self.lstm = nn.LSTM(token_dim, hidden_dim // 2, n_layers, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.crf = CRF(n_tags, -1, pad_id)

    def forward(self, inputs, mask, tags=None):
        """
        Compute predictions or loss if tags is not None.

            -logp(y|x) = exp(f(y,x)) / sum(exp(f(y',x)))

        Args:
            inputs (torch.LongTensor): The sequence input with expected shape (batch_size, seq_len).
            mask (torch.FloatTensor): Batch sequence length with shape (batch_size, seq_len).
            tags (torch.LongTensor): The ground-truth tags with expected shape (batch_size, seq_len).
        """
        embeds = self.token_embeddings(inputs)
        self.lstm.flatten_parameters()
        hidden_states, _ = self.lstm(embeds)
        emissions = self.hidden2tag(hidden_states)
        preds_or_loss = self.crf(emissions, mask, tags, True)

        return preds_or_loss
