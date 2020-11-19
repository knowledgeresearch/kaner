# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""CGN"""

from collections import OrderedDict
from typing import Union
import torch
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from .layers import CRF, GAT, BERTModel


@gctx.register_model("cgn")
class CGN(nn.Module):
    """
    CGN applies graph attention networks on three gazetteer-enhanced graphs to model
    sequence labeling task.

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        n_lexicons (int): The number of lexicons.
        lexicon_dim (int): The dimension of lexicon embedding.
        n_tags (int): The number of tags for labeling sequence.
        hidden_dim (int): The dimension of hidden state.
        n_layers (int): The number of LSTM layers.
        n_gat_layers (int): The number of Graph Attention Layers.
        gat_hidden_dim (int): The dimension of GAT hidden states.
        n_gat_heads (int): The number of attention heads.
        dropout (float): Attention dropout.
        token_embedding_type (str): Token embedding type. Allowed values are 'static' or 'dynamic'. Static means static embeddings,
            including word2vec, glove, etc. Dynamic means dynamic embeddings, including various pretrained language models (BERT).
        token_embeddings (Union[torch.Tensor, OrderedDict]): The pretrained token embeddings, default is None.
        lexicon_embeddings (torch.Tensor): The pretrained lexicon embeddings, default is None.

    References:
        [1] https://github.com/DianboWork/Graph4CNER
        [2] https://www.aclweb.org/anthology/D19-1396/
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
            n_gat_layers: int,
            gat_hidden_dim: int,
            n_gat_heads: int,
            dropout: float,
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
            token_embeddings: Union[torch.Tensor, OrderedDict] = None,
            lexicon_embeddings: torch.FloatTensor = None
    ):
        super(CGN, self).__init__()
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
        self.lexicon_dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(token_dim, hidden_dim // 2, n_layers, batch_first=True, bidirectional=True)
        # let the dimension of lexicon embedding and the dimension of character representation
        # be the same dimension for feeding it to graph attention networks.
        self.lexicon_linear = nn.Linear(lexicon_dim, hidden_dim)
        self.token_linear = nn.Linear(hidden_dim, hidden_dim)
        self.gat_c = GAT(n_gat_layers, n_gat_heads, hidden_dim, gat_hidden_dim, dropout)
        self.gat_t = GAT(n_gat_layers, n_gat_heads, hidden_dim, gat_hidden_dim, dropout)
        self.gat_l = GAT(n_gat_layers, n_gat_heads, hidden_dim, gat_hidden_dim, dropout)
        self.fusion = nn.Linear(hidden_dim + gat_hidden_dim*3, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.crf = CRF(n_tags, -1, pad_id)

    def forward(self, inputs, lexicons, graphs, mask, tags=None):
        """
        Compute predictions or loss if tags is not None.

            -logp(y|x) = exp(f(y,x)) / sum(exp(f(y',x)))

        Args:
            inputs (torch.LongTensor): Character inputs with shape (batch_size, seq_len).
            lexicons (torch.LongTensor): Lexicon inputs with shape (batch_size, lexicon_len).
            graphs (torch.LongTensor): (C-graph, T-gragh, L-graph). Expected shape is (batch_size, 3, n_nodes, n_nodes).
            mask (torch.FloatTensor): Batch sequence length with shape (batch_size, seq_len).
            tags (torch.LongTensor): The ground-truth tags with expected shape (batch_size, seq_len).
        """
        seq_len = inputs.shape[1]
        # (batch_size, seq_len, hidden_dim)
        # Compute character representation via BiLSTM
        if self.token_embedding_type == "static":
            token_embeds = self.token_embeddings(inputs)
        else:
            _, token_embeds = self.bert(inputs)
        self.lstm.flatten_parameters()
        token_features, _ = self.lstm(token_embeds)
        # Compute lexicon representation
        # (batch_size, lexicon_len, hidden_dim)
        lexicon_embeds = self.lexicon_embeddings(lexicons)
        lexicon_features = self.lexicon_dropout(self.lexicon_linear(lexicon_embeds))
        # (batch_size, seq_len + lexicon_len, hidden_dim)
        gat_inputs = torch.cat([token_features, lexicon_features], dim=1)
        # (batch_size, seq_len, hidden_dim)
        gat_outputs_c = self.gat_c(gat_inputs, graphs[:, 0, :, :])[:, :seq_len, :]
        gat_outputs_t = self.gat_t(gat_inputs, graphs[:, 1, :, :])[:, :seq_len, :]
        gat_outputs_l = self.gat_l(gat_inputs, graphs[:, 2, :, :])[:, :seq_len, :]
        new_token_features = self.token_linear(token_features)
        fusion_inputs = torch.cat(
            [new_token_features, gat_outputs_c, gat_outputs_t, gat_outputs_l], dim=-1
        )
        fusion_outputs = self.fusion(fusion_inputs)
        emissions = self.hidden2tag(fusion_outputs)
        preds_or_loss = self.crf(emissions, mask, tags, True)

        return preds_or_loss
