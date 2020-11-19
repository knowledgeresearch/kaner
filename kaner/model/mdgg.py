# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""MDGG"""

from collections import OrderedDict
from typing import Union
import torch
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from .layers import GGNN, CRF, BERTModel


@gctx.register_model("mdgg")
class MDGG(nn.Module):
    """
    The implemention of the paper "A Neural Multi-digraph Model for Chinese NER with Gazetteers".

    References:
        [1] A Neural Multi-digraph Model for Chinese NER with Gazetteers
        [2] https://github.com/PhantomGrapes/MultiDigraphNER

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        n_lexicons (int): The number of lexicons.
        lexicon_dim (int): The dimension of lexicon embedding.
        n_tags (int): The number of tags for labeling sequence.
        hidden_dim (int): The dimension of hidden state.
        n_layers (int): The number of LSTM layers.
        n_edge_types (int): The total number of edge types (character + gazetteer_types).
        n_steps (int): The total number of steps in GGNN.
        dropout (float): Dropout for embeddings.
        token_embedding_type (str): Token embedding type. Allowed values are 'static' or 'dynamic'. Static means static embeddings,
            including word2vec, glove, etc. Dynamic means dynamic embeddings, including various pretrained language models (BERT).
        token_embeddings (Union[torch.Tensor, OrderedDict]): The pretrained token embeddings, default is None.
        lexicon_embeddings (torch.Tensor): The pretrained lexicon embeddings, default is None.
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            n_lexicons: int,
            lexicon_dim: int,
            n_tags: int,
            hidden_dim: int,
            n_layers: int,
            n_edge_types: int,
            n_steps: int,
            pad_id: int,
            dropout: float,
            max_seq_len: int = None,
            n_token_types: int = None,
            layer_norm_eps: float = None,
            hidden_dropout: float = None,
            attn_dropout: float = None,
            n_heads: int = None,
            intermediate_size: int = None,
            n_hidden_layers: int = None,
            token_embedding_type: str = "static",
            token_embeddings: Union[torch.Tensor, OrderedDict] = None,
            lexicon_embeddings: torch.FloatTensor = None
    ):
        super(MDGG, self).__init__()
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
        if lexicon_embeddings is not None:
            self.lexicon_embeddings = nn.Embedding.from_pretrained(lexicon_embeddings)
        else:
            self.lexicon_embeddings = nn.Embedding(n_lexicons, lexicon_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.token_linear = nn.Linear(token_dim, hidden_dim)
        self.lexicon_linear = nn.Linear(lexicon_dim, hidden_dim)
        self.ggnn = GGNN(hidden_dim, n_edge_types, n_steps)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, n_layers, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.crf = CRF(n_tags, -1, pad_id)

    def forward(self, inputs, lexicons, graphs, mask, tags=None):
        """
        Compute predictions or loss if tags is not None.

            -logp(y|x) = exp(f(y,x)) / sum(exp(f(y',x)))

        Args:
            inputs (torch.LongTensor): Character inputs with shape (batch_size, seq_len).
            lexicons (torch.LongTensor): Lexicon inputs with shape (batch_size, lexicon_len).
            graphs (torch.LongTensor): Graph adjacency matrix with shape (batch_size, n_edge_types, n_nodes, n_nodes).
            mask (torch.FloatTensor): Batch sequence length with shape (batch_size, seq_len).
            tags (torch.LongTensor): The ground-truth tags with expected shape (batch_size, seq_len).
        """
        seq_len = inputs.shape[1]
        # (batch_size, seq_len + lexicon_len, hidden_dim)
        if self.token_embedding_type == "static":
            token_embeds = self.dropout(self.token_embeddings(inputs))
        else:
            _, token_embeds = self.bert(inputs)
            token_embeds = self.dropout(token_embeds)
        token_features = self.token_linear(token_embeds)
        lexicon_embeds = self.dropout(self.lexicon_embeddings(lexicons))
        lexicon_features = self.lexicon_linear(lexicon_embeds)
        feats = torch.cat([token_features, lexicon_features], dim=1)
        outputs = self.ggnn(feats, graphs)
        # (batch_size, seq_len, hidden_dim)
        outputs = outputs[:, :seq_len, :]

        # BiLSTM: (batch_size, seq_len, hidden_dim)
        self.lstm.flatten_parameters()
        hidden_states, _ = self.lstm(outputs)
        emissions = self.hidden2tag(hidden_states)
        preds_or_loss = self.crf(emissions, mask, tags, True)

        return preds_or_loss
