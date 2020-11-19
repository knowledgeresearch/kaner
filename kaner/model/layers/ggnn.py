# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Gated Graph Neural Networks (GGNN)"""

import torch
import torch.nn as nn


class Propogator(nn.Module):
    """
    Gated propogator in GGNN with GRU mechanism.

    Reference:
        [1] https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html?highlight=grucell

    Args:
        feat_dim (int): The dimension of features.
        n_edge_types (int): The total number of edge types (character + gazetteer_types).
    """

    def __init__(self, feat_dim: int, n_edge_types: int):
        super(Propogator, self).__init__()
        self.feat_dim = feat_dim
        self.n_edge_types = n_edge_types
        self.gru = nn.GRUCell(feat_dim * n_edge_types, feat_dim)

    def forward(self, a, pre_feats, graphs):
        """
        Compute new feature representations.

        Args:
            a (torch.FloatTensor): Input with shape (n_edge_types, batch_size, n_nodes, feat_dim).
            pre_feats (torch.FloatTensor):: The previous features with shape (batch_size, n_nodes, feat_dim).
            graphs (torch.LongTensor):: Graph adjacency matrix with shape (batch_size, n_edge_types, n_nodes, n_nodes).
        """
        batch_size, n_nodes, feat_dim = pre_feats.shape
        flows = []
        for edge_type_id in range(self.n_edge_types):
            #   (batch_size, n_nodes, n_nodes) * (batch_size, n_nodes, feat_dim)
            # = (batch_size, n_nodes, feat_dim)
            flows.append(torch.bmm(graphs[:, edge_type_id, :, :], a[edge_type_id]))
        # (batch_size, n_nodes, feat_dim * n_edge_types)
        a = torch.cat(flows, dim=2)
        # (batch_size, n_nodes, feat_dim)
        output = self.gru(
            a.view(batch_size * n_nodes, self.n_edge_types * feat_dim),
            pre_feats.view(batch_size * n_nodes, feat_dim)
        )
        output = output.view(batch_size, n_nodes, feat_dim)

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN).

    References:
        [1] https://arxiv.org/abs/1511.05493
        [2] https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/gatedgraphconv.html

    Args:
        feat_dim (int): The dimension of features.
        n_edge_types (int): The total number of edge types (character + gazetteer_types).
        n_steps (int): The total number of steps.
    """

    def __init__(self, feat_dim: int, n_edge_types: int, n_steps: int):
        super(GGNN, self).__init__()
        self.n_edge_types = n_edge_types
        self.n_steps = n_steps
        self.fc = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(n_edge_types)])
        self.propogator = Propogator(feat_dim, n_edge_types)
        self.type_weight = nn.Parameter(torch.tensor([1.0] * n_edge_types))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats, graphs):
        """
        Compute features via GGNN.

        Args:
            feats (torch.FloatTensor): The previous features with shape (batch_size, n_nodes, feat_dim).
            graphs (torch.LongTensor): Graph adjacency matrix with shape (batch_size, n_edge_types, n_nodes, n_nodes).
        """
        weights = self.sigmoid(self.type_weight)
        graphs = torch.cat(
            [(graphs[:, edge_type_id, :, :] * weights[edge_type_id]).unsqueeze(1) for edge_type_id in range(self.n_edge_types)],
            1
        )
        for _ in range(self.n_steps):
            a = []
            for edge_type_id in range(self.n_edge_types):
                a.append(self.fc[edge_type_id](feats))
            feats = self.propogator(a, feats, graphs)

        return feats
