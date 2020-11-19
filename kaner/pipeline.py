# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Training Pipeline"""

from typing import Dict, Any
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from kaner.adapter.in_adapter import split_dataset
from kaner.adapter.out_adapter import BaseOutAdapter, OutMRC
from kaner.adapter.tokenizer import CharTokenizer
from kaner.adapter.knowledge import Gazetteer
from kaner.adapter.mrc import load_query
from kaner.trainer import NERTrainer, TrainerConfig


gctx.init()


def train(config: TrainerConfig) -> Dict[str, Any]:
    """
    Train a model with configuration.

    Args:
        config (TrainerConfig): Trainer Configuration.
    """

    def update_hyperparameters(tokenizer: CharTokenizer, out_adapter: BaseOutAdapter, gazetteer: Gazetteer):
        """
        Update hyper parameters.

        Args:
            tokenizer (CharTokenizer): Tokenizer.
            out_adapter (BaseOutAdapter): Output adapter.
            gazetteer (Gazetteer): Gazetteer.
        """
        partial_configs = {"n_tags": len(out_adapter)}
        partial_configs.update(tokenizer.configs())
        partial_configs.update(gazetteer.configs())

        return partial_configs

    raw_datasets = split_dataset(config.dataset_folder, dataset_pp=config.dataset_pp)
    tokenizer = CharTokenizer(config.tokenizer_model_folder)
    tokenizer.save(config.output_folder)
    gazetteer = Gazetteer(config.gazetteer_model_folder)
    gazetteer.save(config.output_folder)
    if config.mrc_mode:
        out_adapter = OutMRC()
        queries = load_query(config.dataset_folder, "query.csv")
    else:
        out_adapter = gctx.create_outadapter(config.out_adapter, dataset_folder=config.dataset_folder, file_name="labels")
        out_adapter.save(config.output_folder, "labels")
        queries = None
    for raw_dataset in raw_datasets:
        gazetteer.count_freq(raw_dataset)
    in_adapters = (
        gctx.create_inadapter(
            config.in_adapter, dataset=dataset, tokenizer=tokenizer, out_adapter=out_adapter, gazetteer=gazetteer, queries=queries,
            **config.hyperparameters
        )
        for dataset in raw_datasets
    )
    token_embeddings = tokenizer.embeddings()
    lexicon_embeddings = gazetteer.embeddings()
    config.hyperparameters = update_hyperparameters(tokenizer, out_adapter, gazetteer)
    collate_fn = gctx.create_batcher(
        config.model, input_pad=tokenizer.pad_id, output_pad=out_adapter.unk_id, lexicon_pad=gazetteer.pad_id, device=config.device
    )
    model = gctx.create_model(config.model, **config.hyperparameters, token_embeddings=token_embeddings, lexicon_embeddings=lexicon_embeddings)
    trainer = NERTrainer(
        config, tokenizer, in_adapters, out_adapter, collate_fn, model, nn.CrossEntropyLoss(),
        gctx.create_traincallback(config.model), gctx.create_testcallback(config.model)
    )
    results = trainer.train()

    return results, trainer
