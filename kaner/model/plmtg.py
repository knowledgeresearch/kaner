# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Pretrained Language Model based Tagger"""

from collections import OrderedDict
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from .layers import BERTModel


def _update_weights_by_n_layers(weights: OrderedDict, n_hidden_layers: int) -> OrderedDict:
    """
    Update layers by the number of layers.

    Args:
        weights (OrderedDict): The pretrained weights.
        n_hidden_layers (int): The number of hidden layers to be loaded.
    """
    assert isinstance(n_hidden_layers, int) and n_hidden_layers > 0
    valid_layers = ["encoder.layers.{0}.".format(i) for i in range(n_hidden_layers)]
    new_weights = OrderedDict()
    for key, value in weights.items():
        if key.startswith("encoder.layers."):
            for valid_layer in valid_layers:
                if key.startswith(valid_layer):
                    new_weights[key] = value
        else:
            new_weights[key] = value

    return new_weights


@gctx.register_model("plmtg")
class PLMTG(nn.Module):
    """
    A Pretrained Language Model based Tagger for sequence labeling.

    Args:
        n_tokens (int): The number of tokens.
        token_dim (int): The dimension of token embedding.
        max_seq_len (int): The maximum length of a sequence.
        n_token_types (int): The number of token types.
        pad_id (int): PAD id.
        n_heads (int): The number of attention heads.
        intermediate_size (int): The size of FeedForward intermediate.
        n_tags (int): The number of tags.
        n_layers (int): The number of LSTM layers.
        hidden_dim (int): The dimension of hidden state.
        attn_dropout (float): Attention dropout probability.
        layer_norm_eps (float): Layer normalization epsilon.
        hidden_dropout (float): Hidden dropout probability.
        token_embeddings (OrderedDict): The pretrained weights of the language model.
    """

    def __init__(
            self,
            n_tokens: int,
            token_dim: int,
            max_seq_len: int,
            n_token_types: int,
            n_heads: int,
            intermediate_size: int,
            n_hidden_layers: int,
            n_tags: int,
            hidden_dim: int,
            pad_id: int,
            attn_dropout: float,
            layer_norm_eps: float,
            hidden_dropout: float,
            token_embeddings: OrderedDict = None
    ):
        super(PLMTG, self).__init__()
        assert n_hidden_layers > 0
        self.plm_encoder = BERTModel(
            n_tokens, token_dim, max_seq_len, n_token_types, pad_id, n_heads, hidden_dim,
            intermediate_size, n_hidden_layers, attn_dropout, layer_norm_eps, hidden_dropout
        )
        if token_embeddings is not None:
            self.plm_encoder.load_state_dict(_update_weights_by_n_layers(token_embeddings, n_hidden_layers))
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)

    def forward(self, inputs, mask=None):
        """
        Return the loggits of each token encoded by pretrained language model.

        Args:
            inputs (torch.LongTensor): The sequence input with expected shape (batch_size, seq_len).
            mask (torch.ByteTensor): Masks with expected shape (batch_size, seq_len).
        """
        _, hidden_states = self.plm_encoder(inputs, mask)
        logits = self.hidden2tag(hidden_states)

        return logits
