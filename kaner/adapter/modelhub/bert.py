# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Modelhub: BERT"""

import os
from collections import OrderedDict
from typing import Dict, Union

import torch

from kaner.context import GlobalContext as gctx
from kaner.common import load_json
from kaner.model.layers import BERTModel
from .base import BaseModelhub


@gctx.register_tokenizer("bert")
class BERT(BaseModelhub):
    """
    BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

    References:
        [1] https://arxiv.org/abs/1810.04805

    Args:
        root_folder (str): The root folder of the dataset.
    """
    file_names = [
        "tokens.txt", "token_embeddings.checkpoints", "token_configs.json"
    ]

    def __init__(self, root_folder: str):
        super(BERT, self).__init__(
            root_folder, "BERT",
            [
                "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json",
                "https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin",
                "https://cdn.huggingface.co/bert-base-chinese-vocab.txt"
            ]
        )

    def _preprocess(self) -> Dict[str, Union[list, torch.FloatTensor, dict]]:
        """
        Preprocess the model.
        """
        results = {}
        raw_folder = os.path.join(self.root_folder, "raw")
        # tokens
        with open(os.path.join(raw_folder, "bert-base-chinese-vocab.txt"), "r", encoding="utf-8") as f_in:
            tokens = [line.replace("\n", "") for line in f_in.readlines()]
        results["tokens.txt"] = tokens
        # configurations
        config = load_json("utf-8", raw_folder, "bert-base-chinese-config.json")
        new_config = {
            "n_tokens": config["vocab_size"],
            "token_dim": config["hidden_size"],
            "max_seq_len": config["max_position_embeddings"],
            "n_token_types": config["type_vocab_size"],
            "pad_token_id": config["pad_token_id"],
            "n_heads": config["num_attention_heads"],
            "hidden_dim": config["hidden_size"],
            "intermediate_size": config["intermediate_size"],
            "n_hidden_layers": config["num_hidden_layers"],
            "attn_dropout": config["attention_probs_dropout_prob"],
            "layer_norm_eps": config["layer_norm_eps"],
            "hidden_dropout": config["hidden_dropout_prob"],
            "token_embedding_type": "dynamic"
        }
        results["token_configs.json"] = new_config
        # weights
        weights = torch.load(os.path.join(raw_folder, "bert-base-chinese-pytorch_model.bin"))
        new_weights = OrderedDict()
        for key, value in weights.items():
            if key.startswith("bert"):
                key = key.replace("bert.", "")
                key = key.replace("layer", "layers")
                key = key.replace("word_embeddings", "token_embeddings")
                key = key.replace("LayerNorm.gamma", "layernorm.weight")
                key = key.replace("LayerNorm.beta", "layernorm.bias")
                key = key.replace("intermediate.dense.", "intermediate.dense.0.")
                key = key.replace("self", "attention")
                for layer_id in range(new_config["n_hidden_layers"]):
                    key = key.replace(
                        "encoder.layers.{0}.output.dense.".format(layer_id), "encoder.layers.{0}.intermediate.dense.2.".format(layer_id)
                    )
                    key = key.replace(
                        "encoder.layers.{0}.output.layernorm.".format(layer_id), "encoder.layers.{0}.intermediate.layernorm.".format(layer_id)
                    )
                new_weights[key] = value
        new_weights["embeddings.position_ids"] = torch.arange(new_config["max_seq_len"]).expand((1, -1))
        results["token_embeddings.checkpoints"] = new_weights
        assert set(new_weights.keys()) == set(BERTModel(**new_config).state_dict().keys())

        return results
