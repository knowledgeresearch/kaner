# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""BERT"""

import torch
import torch.nn as nn
from .attention import BERTAttention


class BERTEmbedding(nn.Module):
    """
    BERTEmbedding consists of token, position and token_type embeddings.

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        max_seq_len (int): The maximum length of a sequence.
        n_token_types (int: The number of token types.
        pad_token_id (int): PAD ID.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): The embedding dropout probability.
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            max_seq_len: int,
            n_token_types: int,
            pad_token_id: int,
            layer_norm_eps: float,
            hidden_dropout: float
    ):
        super(BERTEmbedding, self).__init__()
        self.token_embeddings = nn.Embedding(n_tokens, token_dim, pad_token_id)
        self.position_embeddings = nn.Embedding(max_seq_len, token_dim)
        self.token_type_embeddings = nn.Embedding(n_token_types, token_dim)
        self.layernorm = nn.LayerNorm(token_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)
        self.register_buffer("position_ids", torch.arange(max_seq_len).expand((1, -1)))

    def forward(self, inputs, position_ids=None, token_type_ids=None):
        """
        Return BERT embeddings.

        Args:
            inputs (torch.LongTensor): Input with shape (batch_size, seq_len).
            position_ids (torch.LongTensor): Position ids with shape (batch_size, seq_len).
            token_type_ids (torch.LongTensor): Token type ids with shape (batch_size, seq_len).
        """
        seq_len = inputs.shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = torch.zeros(inputs.shape, dtype=torch.long, device=self.position_ids.device)

        token_embeds = self.token_embeddings(inputs)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Intermediate(nn.Module):
    """
    Intermediate (also known as FeedForward) creates a new representation (via nonlinear transform, as opposed
    to the self attention layer) based on the information from the self attention layer - which is now
    "contextualized embeddings".

    References:
        [1] https://www.reddit.com/r/MachineLearning/comments/bnejs3/d_what_does_the_feedforward_neural_network_in/

    Args:
        hidden_dim (int): The dimension of hidden states.
        intermediate_size (int): The intermediate size of feedforword layer.
        layer_norm_eps (float): The layer normalization epsilon.
        hidden_dropout (float): The hidden dropout probability.
    """
    def __init__(self, hidden_dim: int, intermediate_size: int, hidden_dropout: float, layer_norm_eps: float):
        super(Intermediate, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, inputs):
        """
        Return the context-embeddings.

        Args:
            inputs (torch.FloatTensor): Attention outputs with shape (batch_size, seq_len, hidden_dim).
        """
        hidden_states = self.dense(inputs)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + inputs)

        return hidden_states


class BERTLayer(nn.Module):
    """
    BERTLayer define the main computation of a BERT layer.

    Args:
        n_heads (int): The number of attention heads.
        hidden_dim (int): The dimension of hidden states.
        intermediate_size (int): The intermediate size of feedforword layer.
        attn_dropout (float): The attention dropout probability.
        layer_norm_eps (float): The layer normalization epsilon.
        hidden_dropout (float): The hidden dropout probability.
    """

    def __init__(
            self,
            n_heads: int,
            hidden_dim: int,
            intermediate_size: int,
            attn_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float
    ):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(n_heads, hidden_dim, attn_dropout, layer_norm_eps, hidden_dropout)
        self.intermediate = Intermediate(hidden_dim, intermediate_size, hidden_dropout, layer_norm_eps)

    def forward(self, hidden_states, mask=None):
        """
        Return the context-embeddings.

        Args:
            hidden_states (torch.FloatTensor): Hidden states with shape (batch_size, seq_len, hidden_dim).
            mask (torch.LongTensor): Masks with shape (batch_size, seq_len).
        """
        outputs = self.attention(hidden_states, mask)
        outputs = self.intermediate(outputs)

        return outputs


class BERTEncoder(nn.Module):
    """
    BERTEncoder consists of n-layers of transformer layer for compute context-aware features.

    Args:
        n_heads (int): The number of attention heads.
        hidden_dim (int): The dimension of hidden states.
        intermediate_size (int): The intermediate size of feedforword layer.
        n_hidden_layers (int): The number of hidden layer in BERT.
        attn_dropout (float): The attention dropout probability.
        layer_norm_eps (float): The layer normalization epsilon.
        hidden_dropout (float): The hidden dropout probability.
    """
    def __init__(
            self,
            n_heads: int,
            hidden_dim: int,
            intermediate_size: int,
            n_hidden_layers: int,
            attn_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float
    ):
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                BERTLayer(n_heads, hidden_dim, intermediate_size, attn_dropout, layer_norm_eps, hidden_dropout)
                for _ in range(n_hidden_layers)
            ]
        )

    def forward(self, hidden_states, mask=None):
        """
        Return the context-embeddings.

        Args:
            hidden_states (torch.FloatTensor): Hidden states with shape (batch_size, seq_len, hidden_dim).
            mask (torch.LongTensor): Masks with shape (batch_size, seq_len).
        """
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, mask)

        return hidden_states


class BERTPooler(nn.Module):
    """
    Pool the output of first token of BERT.

    Args:
        hidden_dim (int): The dimension of hidden states.
    """

    def __init__(self, hidden_dim: int):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        Return the pooled output of the first token representation.

        Args:
            hidden_states (torch.FloatTensor): Hidden states with shape (batch_size, seq_len, hidden_dim).
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BERTModel(nn.Module):
    """
    The implementation of the paper `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`.

    References:
        [1] https://arxiv.org/abs/1810.04805
        [2] https://github.com/huggingface/transformers

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        max_seq_len (int): The maximum length of sequence.
        n_token_types (int): The number of token types.
        pad_token_id (int): PAD id.
        n_heads (int): The number of attention heads.
        hidden_dim (int): The dimension of hidden states.
        intermediate_size (int): The intermediate size of feedforword layer.
        n_hidden_layers (int): The number of hidden layer in BERT.
        attn_dropout (float): The attention dropout probability.
        layer_norm_eps (float): The layer normalization epsilon.
        hidden_dropout (float): The hidden dropout probability.
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            max_seq_len: int,
            n_token_types: int,
            pad_token_id: int,
            n_heads: int,
            hidden_dim: int,
            intermediate_size: int,
            n_hidden_layers: int,
            attn_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float,
            *args,
            **kwargs
    ):
        super(BERTModel, self).__init__()
        self.embeddings = BERTEmbedding(
            n_tokens, token_dim, max_seq_len, n_token_types, pad_token_id, layer_norm_eps, hidden_dropout
        )
        self.encoder = BERTEncoder(
            n_heads, hidden_dim, intermediate_size, n_hidden_layers, attn_dropout, layer_norm_eps, hidden_dropout
        )
        self.pooler = BERTPooler(hidden_dim)

    def forward(self, inputs, mask=None, position_ids=None, token_type_ids=None):
        """
        Compute dynamic token representation according to a context by BERT model.

        Args:
            inputs (torch.LongTensor): Inputs with shape (batch_size, seq_len).
            mask (torch.LongTensor): Masks with shape (batch_size, seq_len).
            position_ids (torch.LongTensor): Position type ids with shape (batch_size, seq_len).
            token_type_ids (torch.LongTensor): Token type ids with shape (batch_size, seq_len).
        """
        embedding_output = self.embeddings(inputs, position_ids, token_type_ids)
        encoder_outputs = self.encoder(embedding_output, mask)
        pooled_output = self.pooler(encoder_outputs)

        return (pooled_output, encoder_outputs)
