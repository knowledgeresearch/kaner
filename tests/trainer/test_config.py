# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Trainer Configuration Tests"""

from copy import deepcopy

from kaner.trainer.config import TrainerConfig


def test_config():
    """Test the class `TrainerConfig`."""
    configs = {
        "task": "NER",
        "model": "blcrf",
        "dataset": "weiboner",
        "in_adapter": "in_seqlab",
        "out_adapter": "out_seqlab",
        "data_folder": "./data",
        "tokenizer_model": "gigaword",
        "gazetteer_model": "sgns",
        "gpu": [0],
        "max_seq_len": 512,
        "epoch": 128,
        "batch_size": 256,
        "early_stop_loss": 0.01,
        "stop_if_no_improvement": 10,
        "optim": {
            "optimizer": "Adam",
            "lr_scheduler": "LambdaLR",
            "lr": 0.01
        },
        "hyperparameters": {
            "n_tokens": 11329,
            "token_dim": 50,
            "n_tags": 17,
            "n_layers": 4,
            "hidden_dim": 300,
            "n_lexicons": 704369,
            "lexicon_dim": 50,
            "n_edge_types": 2
        }
    }
    new_configs = deepcopy(configs)
    ins = TrainerConfig(configs, dataset="msraner")
    assert ins.task == "NER"
    assert ins.model == "blcrf"
    assert ins.dataset == "msraner"
    assert ins.dataset_folder == "./data/datahub/msraner"
    assert ins.in_adapter == "in_seqlab"
    assert ins.out_adapter == "out_seqlab"
    assert ins.tokenizer_model == "gigaword"
    assert ins.gazetteer_model == "sgns"
    assert ins.gpu == [0]
    assert ins.max_seq_len == 512
    assert ins.batch_size == 256
    assert ins.early_stop_loss == 0.01
    assert ins.optim == configs["optim"]
    assert ins.hyperparameters == new_configs["hyperparameters"]
    new_configs["hyperparameters"]["example"] = 100
    ins.hyperparameters = {"example": 100}
    assert ins.hyperparameters == new_configs["hyperparameters"]
