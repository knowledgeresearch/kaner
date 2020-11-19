# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Base Trainer"""

import os
import logging
import json
from copy import deepcopy
from typing import Tuple, Callable, Dict, Union, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kaner.common import save_json
from kaner.common.func import timing, feed_args
from kaner.adapter.tokenizer import BaseTokenizer
from kaner.adapter.in_adapter import BaseInAdapter
from kaner.adapter.out_adapter import BaseOutAdapter
from .config import TrainerConfig


class BaseTrainer:
    """
    BaseTriner provides some general utils for training models. Besides, it provides
    some interfaces to thoese subclasses to implement, such as train, and test.

    Args:
        config (TrainerConfig): Trainer configuration.
        tokenizer (BaseTokenizer): Tokenizer.
        in_adapters (Tuple[BaseInAdapter, BaseInAdapter, BaseInAdapter]): Train/Dev/Test input adapters.
        out_adapter (BaseOutAdapter): Output adapter.
        collate_fn (Callable[[tuple], tuple]): Collate function for preprocessing batch data.
        model (nn.Module): Model used to be learned.
        loss_fn (nn.Module): Loss function.
    """
    def __init__(
            self,
            config: TrainerConfig,
            tokenizer: BaseTokenizer,
            in_adapters: Tuple[BaseInAdapter, BaseInAdapter, BaseInAdapter],
            out_adapter: BaseOutAdapter,
            collate_fn: Callable[[tuple], tuple],
            model: nn.Module,
            loss_fn: nn.Module
    ):
        super(BaseTrainer, self).__init__()
        # initailize folders and log module
        if not os.path.isdir(config.output_folder):
            os.makedirs(config.output_folder)
        writer_folder = os.path.join(config.output_folder, "summary")
        if not os.path.isdir(writer_folder):
            os.makedirs(writer_folder)
        self._logger = logging.getLogger(__name__)
        self._logger.handlers.clear()
        ch = logging.StreamHandler()
        fl = logging.FileHandler(os.path.join(config.output_folder, "log.txt"))
        formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        fl.setFormatter(formatter)
        self._logger.addHandler(fl)
        self._logger.addHandler(ch)
        self._logger.setLevel(logging.DEBUG)
        self._writer = SummaryWriter(writer_folder)
        self._config = config
        # initialize adapters and dataloaders
        self._tokenizer = tokenizer
        self._train_in_adapter, self._dev_in_adapter, self._test_in_adapter = in_adapters
        self._collate_fn = collate_fn
        self._train_loader = DataLoader(self._train_in_adapter, config.batch_size, collate_fn=collate_fn)
        self._dev_loader = DataLoader(self._dev_in_adapter, self._config.batch_size, collate_fn=collate_fn)
        self._test_loader = DataLoader(self._test_in_adapter, self._config.batch_size, collate_fn=collate_fn)
        self._out_adapter = out_adapter
        # initialize model and optimizer
        self._model = model
        self._model = self._model.to(self._config.device)
        if len(self._config.gpu) > 1:
            self._model = DataParallel(self._model, device_ids=self._config.gpu)
        self._loss_fn = loss_fn
        self._optimizer, self._lr_scheduler = self._set_optim(config.optim, model)
        save_json(config.to_dict(), config.output_folder, "config.json")

    def _set_optim(self, options: Dict[str, Any], model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Instantiate optimizer and lr_scheduler by a given name.

        Args:
            options (Dict[str, Any]): Avaiable options for instantiating optmizers.
            model (nn.Module): Model to be trained.
        """
        options = deepcopy(options)
        avaiable_optimizers = ["Adam", "SGD"]
        if options["optimizer"] not in avaiable_optimizers:
            print("[Optimizer Error] Avaiable optimizers:", avaiable_optimizers)
            print("https://pytorch.org/docs/stable/optim.html")
            exit(0)
        avaiable_lr_schedulers = ["LambdaLR", "CosineAnnealingWarmRestarts"]
        if options["lr_scheduler"] not in avaiable_lr_schedulers:
            print("[LR Scheduler Error] Avaiable lr_schedulers:", avaiable_lr_schedulers)
            print("https://pytorch.org/docs/stable/optim.html")
            exit(0)
        param_groups = []
        if "lr_group" in options.keys():
            group_names = list(options["lr_group"].keys())
            param_groups.append(
                {
                    "params": [
                        kv[1] for kv in list(
                            filter(lambda kv: all([not kv[0].startswith(name) for name in group_names]), model.named_parameters())
                        )
                    ]
                }
            )
            for group_name in group_names:
                param_groups.append(
                    {
                        "params": [kv[1] for kv in list(filter(lambda kv: kv[0].startswith(group_name), model.named_parameters()))],
                        "lr": options["lr_group"][group_name]
                    }
                )
        else:
            param_groups.append({"params": list(model.parameters())})
        optimizer_class = optim.__dict__[options["optimizer"]]
        optimizer = optimizer_class(param_groups, **feed_args(optimizer_class.__init__, options))
        options.pop("optimizer")
        if "lr_lambda" in options.keys():
            exec("lr_lambda = " + options["lr_lambda"])
            options["lr_lambda"] = locals()["lr_lambda"]
        lr_scheduler_class = optim.lr_scheduler.__dict__[options["lr_scheduler"]]
        lr_scheduler = lr_scheduler_class(optimizer=optimizer, **feed_args(lr_scheduler_class.__init__, options))

        return optimizer, lr_scheduler

    def load_checkpoint(self, checkpoint_path: str = None) -> bool:
        """
        Load checkpoint from the local disk.

        Args:
            checkpoint_path (str): The checkpoint path of the current model.
        """
        if checkpoint_path is None and not isinstance(checkpoint_path, str):
            file_name = "{0}_{1}.checkpoints".format(self._config.model, self._config.dataset)
            checkpoint_path = os.path.join(self._config.output_folder, file_name)
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self._config.device))
            if isinstance(self._model, DataParallel):
                self._model.module.load_state_dict(checkpoint)
            elif isinstance(self._model, nn.Module):
                self._model.load_state_dict(checkpoint)
            return True
        self.log("failed to load model's checkpoint: {0}, you may need to train a model.".format(checkpoint_path))

        return False

    def save_checkpoint(self, checkpoint_path: str = None) -> None:
        """
        Save checkpoint into the local disk.

        Args:
            checkpoint_path (str): The checkpoint path of the current model.
        """
        if checkpoint_path is None and not isinstance(checkpoint_path, str):
            file_name = "{0}_{1}.checkpoints".format(self._config.model, self._config.dataset)
            checkpoint_path = os.path.join(self._config.output_folder, file_name)
        if isinstance(self._model, DataParallel):
            checkpoint = self._model.module.state_dict()
            torch.save(checkpoint, checkpoint_path)
        else:
            checkpoint = self._model.state_dict()
            torch.save(checkpoint, checkpoint_path)

    def log(self, content: Union[str, dict], x_value: float = None, y_value: float = None) -> None:
        """
        Record status by logging, tensorboard. If x_value or y_value is None, we regard content
        as text and record it. Otherwise, we regard content as a record classification for merge
        the same values (x, y). The commonly used record classifications are F1 score, precision
        score, etc.

        Args:
            content (Union[str, dict]): Logging content.
            x_value (float): X input.
            y_value (float): Y input.
        """
        if x_value is None or y_value is None:
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            assert isinstance(content, str)
            self._writer.add_text(
                "{0}/{1}/{2}/texts".format(self._config.identity, self._config.model, self._config.dataset), content
            )
            content = "[{0}, {1}] {2}".format(self._config.model, self._config.dataset, content)
            self._logger.debug(content)
        else:
            self._writer.add_scalar(
                "{0}/{1}/{2}/{3}".format(self._config.identity, self._config.model, self._config.dataset, content),
                y_value, x_value
            )

    def _train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        raise NotImplementedError

    @timing
    def train(self) -> Dict[str, Any]:
        """
        Train a model and return the results.
        """
        results = self._train()

        return results

    def _test(self, loader: DataLoader) -> Dict[str, Any]:
        """
        Test a model and return the results.

        Args:
            loader (DataLoader): The dataloader used to be tested.
        """
        raise NotImplementedError

    def startp(self, checkpoint_path: str = None) -> None:
        """
        Prepare for predicting texts, including loading model and error handling.

        Args:
            checkpoint_path (str): The checkpoint path of the current model.
        """
        if not self.load_checkpoint(checkpoint_path):
            exit(0)
        self._model.eval()

    def predict(self, texts: List[str]) -> List[dict]:
        """
        Given a list of texts, predict their spans using the current model.
        """
        raise NotImplementedError
