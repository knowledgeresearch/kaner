# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""GAT Tests"""

import torch
from kaner.model.layers.gat import GraphAttentionLayer


def test_gat():
    """Test the module `gat`."""
    gat_layer = GraphAttentionLayer(2, 3, 0)
    gat_layer.w.data = torch.ones(gat_layer.w.data.shape)
    gat_layer.a1.data = torch.ones(gat_layer.a1.data.shape)
    gat_layer.a2.data = torch.ones(gat_layer.a2.data.shape)
    # graph: 1 <-> 2 <-> 3
    graphs = torch.tensor([[[1, 1, 0], [1, 1, 1], [0, 1, 1]]])
    features = torch.tensor([[[0.1, 0.1], [0.5, 0.5], [1.0, 1.0]]])
    outputs = gat_layer(features, graphs)
    true_outputs = [
        [
            [0.9334617853164673, 0.9334617853164673, 0.9334617853164673],
            [1.945066213607788, 1.945066213607788, 1.945066213607788],
            [1.9525741338729858, 1.9525741338729858, 1.9525741338729858]
        ]
    ]
    assert outputs.tolist() == true_outputs
