# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Callbacks: train/_test"""

import torch
import torch.nn as nn

from kaner.context import GlobalContext as gctx
from kaner.adapter.out_adapter import BaseOutAdapter
from kaner.adapter.batcher import batch_seq_mask
from kaner.trainer import TrainerConfig


@gctx.register_traincallback("blcrf")
def train_callback_blcrf(batch_group: tuple, model: nn.Module, loss_fn: nn.Module, config: TrainerConfig) -> torch.FloatTensor:
    """
    Callback of the function train for the model BLCRF.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        loss_fn (nn.Module): Loss function.
        config (TrainerConfig): Trainer configuration.
    """
    _, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    loss = model(ready_to_eat["inputs"], 1 - mask, ready_to_eat["outputs"])
    if len(config.gpu) > 1:
        loss = loss.mean()

    return loss


@gctx.register_testcallback("blcrf")
def test_callback_blcrf(batch_group: tuple, model: nn.Module, out_adapter: BaseOutAdapter, config: TrainerConfig) -> tuple:
    """
    Callback of the function test for the model BLCRF.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        out_adapter (BaseOutAdapter): The output adapter that converts tensors to raw data.
        config (TrainerConfig): Trainer configuration.
    """
    batch, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    predictions = model(ready_to_eat["inputs"], 1 - mask)
    lengths = ready_to_eat["lengths"].tolist()
    true_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(ready_to_eat["outputs"].tolist())]
    pred_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(predictions.tolist())]

    return (true_tags, pred_tags)


@gctx.register_traincallback("plmtg")
def train_callback_plmtg(batch_group: tuple, model: nn.Module, loss_fn: nn.Module, config: TrainerConfig) -> torch.FloatTensor:
    """
    Callback of the function train for the model PLMTG.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        loss_fn (nn.Module): Loss function.
        config (TrainerConfig): Trainer configuration.
    """
    _, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device)
    logits = model(ready_to_eat["inputs"], mask)
    loss = loss_fn(logits.view(-1, logits.shape[-1]), ready_to_eat["outputs"].view(-1))

    return loss


@gctx.register_testcallback("plmtg")
def test_callback_plmtg(batch_group: tuple, model: nn.Module, out_adapter: BaseOutAdapter, config: TrainerConfig) -> tuple:
    """
    Callback of the function test for the model PLMTG.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        out_adapter (BaseOutAdapter): The output adapter that converts tensors to raw data.
        config (TrainerConfig): Trainer configuration.
    """
    batch, ready_to_eat = batch_group
    predictions = model(ready_to_eat["inputs"])
    lengths = ready_to_eat["lengths"].tolist()
    true_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(ready_to_eat["outputs"].tolist())]
    pred_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(predictions.argmax(-1).tolist())]

    return (true_tags, pred_tags)


@gctx.register_traincallback("ses")
def train_callback_ses(batch_group: tuple, model: nn.Module, loss_fn: nn.Module, config: TrainerConfig) -> torch.FloatTensor:
    """
    Callback of the function train for the model SES.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        loss_fn (nn.Module): Loss function.
        config (TrainerConfig): Trainer configuration.
    """
    _, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    loss = model(ready_to_eat["inputs"], ready_to_eat["lexicons"], ready_to_eat["weights"], 1 - mask, ready_to_eat["outputs"])
    if len(config.gpu) > 1:
        loss = loss.mean()

    return loss


@gctx.register_testcallback("ses")
def test_callback_ses(batch_group: tuple, model: nn.Module, out_adapter: BaseOutAdapter, config: TrainerConfig) -> tuple:
    """
    Callback of the function test for the model SES.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        out_adapter (BaseOutAdapter): The output adapter that converts tensors to raw data.
        config (TrainerConfig): Trainer configuration.
    """
    batch, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    predictions = model(ready_to_eat["inputs"], ready_to_eat["lexicons"], ready_to_eat["weights"], 1 - mask)
    lengths = ready_to_eat["lengths"].tolist()
    true_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(ready_to_eat["outputs"].tolist())]
    pred_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(predictions.tolist())]

    return (true_tags, pred_tags)


@gctx.register_traincallback("cgn")
def train_callback_cgn(batch_group: tuple, model: nn.Module, loss_fn: nn.Module, config: TrainerConfig) -> torch.FloatTensor:
    """
    Callback of the function train for the model CGN.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        loss_fn (nn.Module): Loss function.
        config (TrainerConfig): Trainer configuration.
    """
    _, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    loss = model(ready_to_eat["inputs"], ready_to_eat["lexicons"], ready_to_eat["graphs"], 1 - mask, ready_to_eat["outputs"])
    if len(config.gpu) > 1:
        loss = loss.mean()

    return loss


@gctx.register_testcallback("cgn")
def test_callback_cgn(batch_group: tuple, model: nn.Module, out_adapter: BaseOutAdapter, config: TrainerConfig) -> tuple:
    """
    Callback of the function test for the model CGN.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        out_adapter (BaseOutAdapter): The output adapter that converts tensors to raw data.
        config (TrainerConfig): Trainer configuration.
    """
    batch, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    predictions = model(ready_to_eat["inputs"], ready_to_eat["lexicons"], ready_to_eat["graphs"], 1 - mask)
    lengths = ready_to_eat["lengths"].tolist()
    true_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(ready_to_eat["outputs"].tolist())]
    pred_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(predictions.tolist())]

    return (true_tags, pred_tags)


@gctx.register_traincallback("mdgg")
def train_callback_mdgg(batch_group: tuple, model: nn.Module, loss_fn: nn.Module, config: TrainerConfig) -> torch.FloatTensor:
    """
    Callback of the function train for the model MDGG.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        loss_fn (nn.Module): Loss function.
        config (TrainerConfig): Trainer configuration.
    """
    _, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    loss = model(ready_to_eat["inputs"], ready_to_eat["lexicons"], ready_to_eat["graphs"], 1 - mask, ready_to_eat["outputs"])
    if len(config.gpu) > 1:
        loss = loss.mean()

    return loss


@gctx.register_testcallback("mdgg")
def test_callback_mdgg(batch_group: tuple, model: nn.Module, out_adapter: BaseOutAdapter, config: TrainerConfig) -> tuple:
    """
    Callback of the function test for the model MDGG.

    Args:
        batch_group (tuple): Batch tensor data.
        model (nn.Module): Model to be trained.
        out_adapter (BaseOutAdapter): The output adapter that converts tensors to raw data.
        config (TrainerConfig): Trainer configuration.
    """
    batch, ready_to_eat = batch_group
    mask = torch.tensor(batch_seq_mask(ready_to_eat["lengths"].tolist())).to(ready_to_eat["inputs"].device).to(torch.uint8)
    predictions = model(ready_to_eat["inputs"], ready_to_eat["lexicons"], ready_to_eat["graphs"], 1 - mask)
    lengths = ready_to_eat["lengths"].tolist()
    true_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(ready_to_eat["outputs"].tolist())]
    pred_tags = [[out_adapter[tag_id] for tag_id in sequence[batch[i]["start"]:lengths[i]]] for i, sequence in enumerate(predictions.tolist())]

    return (true_tags, pred_tags)
