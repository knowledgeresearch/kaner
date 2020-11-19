# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Attentions
    References:
        [1] https://arxiv.org/pdf/1706.03762.pdf
        [2] https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
        [3] https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
"""

import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention: softmax((QK^T)/sqrt(head_dim))*V

    Args:
        attn_dropout (float): The dropout is used for attention.
    """

    def __init__(self, attn_dropout: float):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Compute attention according to q and k and multiply it with v.

        Args:
            q (torch.FloatTensor): q matrix with shape (batch_size * n_heads, seq_len, head_dim).
            k (torch.FloatTensor): k matrix with shape (batch_size * n_heads, seq_len, head_dim).
            v (torch.FloatTensor): v matrix with shape (batch_size * n_heads, seq_len, head_dim).
            mask (torch.LongTensor): Mask with shape (batch_size * n_heads, seq_len).
        """
        head_dim = k.size(-1)
        attn = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        if mask is not None:
            n_heads = q.shape[0] // mask.shape[0]
            mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            mask = mask.repeat(n_heads, 1, 1, 1).view(attn.shape)
            attn = attn.masked_fill(mask, -math.inf)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention is appealing for the ability to jointly
    attend to information from different representation subspaces
    at different positions.

    Args:
        n_heads (int): The number of heads.
        hidden_dim (int): The dimension of input features.
        attn_dropout (float): The dropout is used for attention.
    """

    def __init__(self, n_heads: int, hidden_dim: int, attn_dropout: float):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = int(hidden_dim / n_heads)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention = ScaledDotProductAttention(attn_dropout)
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, inputs, mask=None):
        """
        Forward with multi-head attention.

        Args:
            inputs (torch.FloatTensor): The input features with shape (batch_size, seq_len, hidden_dim).
            mask (torch.LongTensor): Mask with shape (batch_size, seq_len).
        """
        n_heads, head_dim = self.n_heads, self.head_dim
        batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        q_m = self.query(inputs).view(batch_size, seq_len, n_heads, head_dim)
        k_m = self.key(inputs).view(batch_size, seq_len, n_heads, head_dim)
        v_m = self.value(inputs).view(batch_size, seq_len, n_heads, head_dim)

        q_m = q_m.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, head_dim)
        k_m = k_m.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, head_dim)
        v_m = v_m.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, head_dim)

        output = self.attention(q_m, k_m, v_m, mask)
        output = output.view(n_heads, batch_size, seq_len, head_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, -1)

        return output


class AttentionOutput(nn.Module):
    """
    AttentionOutput transforms the hidden states from MultiHeadAttention to outputs.

    Args:
        hidden_dim (int): The dimension of input features.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): The hidden states dropout probability.
    """

    def __init__(self, hidden_dim: int, layer_norm_eps: float, hidden_dropout: float):
        super(AttentionOutput, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, outputs, inputs):
        """
        Return the final attention outputs.

        Args:
            outputs (torch.FLoatTensor): Attention outputs.
            inputs (torch.FLoatTensor): Attention inputs.
        """
        hidden_states = self.dense(outputs)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + inputs)

        return hidden_states


class BERTAttention(nn.Module):
    """
    BERT attention with multi-head attention and output transformation.

    Args:
        n_heads (int): The number of heads.
        hidden_dim (int): The dimension of input features.
        attn_dropout (float): The dropout is used for attention.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): The hidden states dropout probability.
    """

    def __init__(
            self,
            n_heads: int,
            hidden_dim: int,
            attn_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float
    ):
        super(BERTAttention, self).__init__()
        self.attention = MultiHeadAttention(n_heads, hidden_dim, attn_dropout)
        self.output = AttentionOutput(hidden_dim, layer_norm_eps, hidden_dropout)

    def forward(self, hidden_states, mask=None):
        """
        Return hidden states processed by bert attentions.

        Args:
            hidden_states (torch.FloatTensor): Inputs with shape (batch_size, seq_len, hidden_dim).
            mask (torch.LongTensor): Mask with shape (batch_size, seq_len).
        """
        outputs = self.attention(hidden_states, mask)
        outputs = self.output(outputs, hidden_states)

        return outputs
