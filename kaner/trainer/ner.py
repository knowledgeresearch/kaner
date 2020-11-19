# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Trainer: Trainer for Named Entity Recognition"""

from typing import Callable, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from kaner.metric.model import compute_ner_prf1
from kaner.adapter.span import to_spans
from kaner.adapter.tokenizer import BaseTokenizer
from kaner.adapter.in_adapter import BaseInAdapter
from kaner.adapter.out_adapter import BaseOutAdapter
from kaner.metric.utils import safe_division
from .base import BaseTrainer, TrainerConfig


class NERTrainer(BaseTrainer):
    """
    Trainer for the task Named Entity Recogtion (NER). More conceptual details can be found in
    https://en.wikipedia.org/wiki/Named-entity_recognition

    Args:
        train_callback (Callable[[tuple, nn.Module, nn.Module], torch.FloatTensor]): Handle the different inputs
            of the different models and return loss.
        test_callback (Callable[[tuple, nn.Module, BaseOutAdapter], tuple]): Handle the different inputs of the
            different models and return the true tags and the prediction tags.
    """

    def __init__(
            self,
            config: TrainerConfig,
            tokenizer: BaseTokenizer,
            in_adapters: Tuple[BaseInAdapter, BaseInAdapter, BaseInAdapter],
            out_adapter: BaseOutAdapter,
            collate_fn: Callable[[tuple], tuple],
            model: nn.Module,
            loss_fn: nn.Module,
            train_callback: Callable[[tuple, nn.Module, nn.Module], torch.FloatTensor],
            test_callback: Callable[[tuple, nn.Module, BaseOutAdapter], tuple]
    ):
        super(NERTrainer, self).__init__(config, tokenizer, in_adapters, out_adapter, collate_fn, model, loss_fn)
        self._train_callback = train_callback
        self._test_callback = test_callback

    def _train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        save_index = -1.0
        no_improvement_epoch, epoch_count = 0, 0
        for epoch_id in range(self._config.epoch):
            total_loss = .0
            self._model.train()
            for batch_group in tqdm.tqdm(self._train_loader, "Epoch {0}".format(epoch_id)):
                self._model.zero_grad()
                loss = self._train_callback(batch_group, self._model, self._loss_fn, self._config)
                loss.backward()
                total_loss += loss.item()
                self._optimizer.step()
                del batch_group, loss
            self._lr_scheduler.step()
            train_loss = total_loss / len(self._train_loader)
            results = self._test(self._dev_loader)
            epoch_count += 1
            if results["f1-score"] > save_index:
                save_index = results["f1-score"]
                no_improvement_epoch = 0
                self.save_checkpoint()
            else:
                no_improvement_epoch += 1
            self.log("dev-f1", epoch_id, results["f1-score"])
            self.log("dev-precision", epoch_id, results["precision-score"])
            self.log("dev-recall", epoch_id, results["recall-score"])
            self.log("dev-loss", epoch_id, results["test-loss"])
            self.log("train-loss", epoch_id, train_loss)
            for group_id, lr in enumerate(self._lr_scheduler.get_last_lr()):
                self.log("lr{0}".format(group_id), epoch_id, lr)
            self.log(
                "epoch: {0}, no_improvement: {1}, dev-f1: {2}, dev-precision: {3}, dev-recall: {4}, dev-loss: {5}, train-loss: {6}".format(
                    epoch_id, no_improvement_epoch,
                    round(results["f1-score"], 5), round(results["precision-score"], 5), round(results["recall-score"], 5),
                    round(results["test-loss"], 5), round(train_loss, 5)
                )
            )
            if train_loss <= self._config.early_stop_loss or (no_improvement_epoch >= self._config.stop_if_no_improvement):
                break
        self.load_checkpoint()
        results = self._test(self._test_loader)
        results["epoch_count"] = epoch_count
        self.log(results)

        return results

    def _test(self, loader: DataLoader) -> Dict[str, Any]:
        """
        Test a model and return the results.

        Args:
            loader (DataLoader): The dataloader used to be tested.
        """
        self._model.eval()
        batch_true_tags, batch_pred_tags = [], []
        test_loss = 0.0
        with torch.no_grad():
            for _, batch_group in enumerate(loader):
                true_tags, pred_tags = self._test_callback(batch_group, self._model, self._out_adapter, self._config)
                batch_true_tags.extend(true_tags)
                batch_pred_tags.extend(pred_tags)
                loss = self._train_callback(batch_group, self._model, self._loss_fn, self._config)
                test_loss += loss.item()
        precision_score, recall_score, f1_score = compute_ner_prf1(batch_pred_tags, batch_true_tags)
        test_loss = safe_division(test_loss, len(loader))
        results = {
            "f1-score": f1_score,
            "precision-score": precision_score,
            "recall-score": recall_score,
            "test-loss": test_loss
        }

        return results

    def predict(self, texts: List[str]) -> List[dict]:
        """
        Given a list of texts, predict their spans using the current model.

        Args:
            texts (List[str]): A list of text to be predicted.
        """
        data = list(self._train_in_adapter.transform_sample({"text": text, "spans": []}) for text in texts)
        batch = self._collate_fn(data)
        with torch.no_grad():
            _, pred_tags = self._test_callback(batch, self._model, self._out_adapter, self._config)
        results = []
        for i, text in enumerate(texts):
            results.append(
                {
                    "text": text,
                    "spans": [vars(span) for span in to_spans(pred_tags[i][:len(text)], list(text), [1.0]*len(text))]
                }
            )

        return results
