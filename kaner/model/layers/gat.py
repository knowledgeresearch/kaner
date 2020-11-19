# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Graph Attention Networks (GAT)"""

import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):
    """
    GraphAttentionLayer applies Graph Attention Networks to input signals of a graph, for
    gathering information of each node from its neighbours.

    Args:
        infeat_dim (int): The dimension of input features.
        outfeat_dim (int): The dimension of output features.
        dropout (float): Attention dropout.
        alpha (float): The negative slope value of LeakyReLU.
        concat (bool): Whether use ELU to activate the output.
    """

    def __init__(self, infeat_dim: int, outfeat_dim: int, dropout: float, alpha: float = 0.1, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.concat = concat
        self.w = nn.Parameter(torch.zeros(infeat_dim, outfeat_dim))
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.a1 = nn.Parameter(torch.zeros(size=(outfeat_dim, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        self.a2 = nn.Parameter(torch.zeros(size=(outfeat_dim, 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.dropout = nn.Dropout(p=dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.elu = torch.nn.ELU()

    def forward(self, features, graph):
        """
        Compute node features of next layer according to graph attention mechanism.

        Args:
            features (torch.FloatTensor): The features of nodes with shape (batch_size, n_nodes, infeat_dim).
            graph (torch.LongTensor): The adjacent matrix of a graph. Expected shape is (batch_size, n_nodes, n_nodes).
        """
        batch_size, n_nodes = features.shape[:2]
        # (batch_size, n_nodes, outfeat_dim)
        hidden = torch.matmul(features, self.w)
        # (batch_size, n_nodes, 1)
        a1 = torch.matmul(hidden, self.a1)
        a2 = torch.matmul(hidden, self.a2)
        # (batch_size, n_nodes, n_nodes)
        e_mat = self.leaky_relu(a1.expand(-1, -1, n_nodes) + a2.expand(-1, -1, n_nodes).transpose(1, 2))
        attn = torch.softmax(e_mat.masked_fill(graph == 0, -1e9), dim=-1)
        attn = self.dropout(attn)
        # (batch_size, n_nodes, outfeat_dim)
        h_prime = torch.matmul(attn, hidden)
        if self.concat:
            return self.elu(h_prime)

        return h_prime


class GAT(nn.Module):
    """
    GAT applies a multi-layer GraphAttentionLayer to process graph data.

    References:
        [1] https://petar-v.com/GAT/
        [2] https://github.com/PetarV-/GAT
        [3] https://arxiv.org/abs/1710.10903
        [4] https://github.com/Diego999/pyGAT
        [5] https://github.com/dsgiitr/graph_nets

    Args:
        n_layers (int): The number of GraphAttentionLayers.
        n_heads (int): The number of attention heads.
        infeat_dim (int): The dimension of input features.
        outfeat_dim (int): The dimension of output features.
        dropout (float): Attention dropout.
    """

    def __init__(self, n_layers: int, n_heads: int, infeat_dim: int, outfeat_dim: int, dropout: float):
        super(GAT, self).__init__()
        assert outfeat_dim % n_heads == 0 and infeat_dim % n_heads == 0 and n_layers > 0
        self.layers = nn.ModuleList([
            nn.ModuleList([GraphAttentionLayer(infeat_dim, infeat_dim//n_heads, dropout) for _ in range(n_heads)])
            for _ in range(n_layers - 1)
        ])
        self.layers.append(GraphAttentionLayer(infeat_dim, outfeat_dim, dropout))

    def forward(self, features, graph):
        """
        Compute node output features via GAT.

        Args:
            features (torch.FloatTensor): The features of nodes with shape (batch_size, n_nodes, infeat_dim).
            graph (torch.LongTensor): The adjacent matrix of a graph. Expected shape is (batch_size, n_nodes, n_nodes).
        """
        for layer_id in range(len(self.layers) - 1):
            features = torch.cat([head(features, graph) for head in self.layers[layer_id]], dim=-1)
        outputs = self.layers[len(self.layers) - 1](features, graph)

        return outputs
