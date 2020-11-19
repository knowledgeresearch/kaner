# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Trainer Configuration"""

import os
from sys import exit
from typing import Dict, Union, Any, List
import torch
from kaner.common import load_yaml_as_json
from kaner.context import GlobalContext as gctx


class TrainerConfig:
    """
    The configuration of trainer.

    Args:
        config_or_configpath (Union[Dict[str, Any], str]): Configuration dictionary or configuration file path.
            It is important to note that the configuration file should be a yaml file.
        dynamic_configs (dict): Dynamic configurations by command arguments which can be used to change
            configurations loaded from the file temporally.
    """
    def __init__(self, config_or_configpath: Union[Dict[str, Any], str], **dynamic_configs):
        super(TrainerConfig, self).__init__()
        if isinstance(config_or_configpath, str) and os.path.isfile(config_or_configpath):
            configs = load_yaml_as_json("utf-8", *(config_or_configpath, ))
        elif isinstance(config_or_configpath, dict):
            configs = config_or_configpath
        else:
            raise ValueError("config_or_configpath mush be dict/str. Value: {0}".format(config_or_configpath))
        for key, value in dynamic_configs.items():
            configs[key] = value
        self._task = configs["task"]
        assert isinstance(self.task, str)
        # indicate whether use MRC mode to recognize entity
        self._mrc_mode = configs.get("mrc_mode", False)
        assert isinstance(self._mrc_mode, bool)
        # initialize dataset
        self._dataset_pp = configs.get("dataset_pp", 1.0)
        assert isinstance(self._dataset_pp, float)
        self._data_folder = configs.get("data_folder", "./data/")
        assert isinstance(self.data_folder, str)
        self._dataset = configs["dataset"]
        assert isinstance(self.dataset, str)
        datasets = gctx.get_dataset_names()
        if self._dataset not in datasets:
            print("The dataset '{0}' is not registered.".format(self.dataset))
            print("Avaiable datasets: {0}".format(datasets))
            exit(0)
        # initialize modelhub
        self._tokenizer_model = configs["tokenizer_model"]
        assert isinstance(self._tokenizer_model, str)
        self._gazetteer_model = configs["gazetteer_model"]
        assert isinstance(self._gazetteer_model, str)
        if self._tokenizer_model not in gctx.get_tokenizer_names():
            print("The tokenizer_model '{0}' is not registered.".format(self._tokenizer_model))
            print("Avaiable modelhub: {0}".format(gctx.get_tokenizer_names()))
            exit(0)
        if self._gazetteer_model not in gctx.get_gazetteer_names():
            print("The gazetteer_model '{0}' is not registered.".format(self._gazetteer_model))
            print("Avaiable modelhub: {0}".format(gctx.get_gazetteer_names()))
            exit(0)
        # initialize model
        self._model = configs["model"]
        assert isinstance(self.model, str)
        if "identity" in configs:
            self._identity = configs["identity"]
        else:
            log_folder = os.path.join(self.data_folder, "logs")
            log_count = 0
            if os.path.isdir(log_folder):
                for file_name in os.listdir(log_folder):
                    if file_name.startswith("trainer-{0}-{1}".format(self.model, self.dataset)):
                        log_count = max(log_count, int(file_name.split("-")[-1]))
            self._identity = "trainer-{0}-{1}-{2}".format(self.model, self.dataset, log_count + 1)
        self._output_folder = os.path.join(self.data_folder, "logs", self._identity)
        self._in_adapter = configs["in_adapter"]
        assert isinstance(self._in_adapter, str)
        self._out_adapter = configs["out_adapter"]
        assert isinstance(self._out_adapter, str)
        self._gpu = configs.get("gpu", [])
        assert isinstance(self.gpu, list) and all([True] + [isinstance(n, int) for n in self.gpu])
        self._device = "cpu"
        self._load_devices()
        self._max_seq_len = configs["hyperparameters"].get("max_seq_len", 512)
        assert isinstance(self._max_seq_len, int) and self.max_seq_len > 0
        # training
        self._epoch = configs.get("epoch", 128)
        assert isinstance(self._epoch, int)
        self._batch_size = configs["batch_size"]
        assert isinstance(self._batch_size, int)
        self._early_stop_loss = configs.get("early_stop_loss", 0.01)
        assert isinstance(self.early_stop_loss, float) and self.early_stop_loss > 0.0
        self._optim = configs["optim"]
        assert isinstance(self._optim, dict)
        self._stop_if_no_improvement = configs.get("stop_if_no_improvement", 10)
        assert isinstance(self._stop_if_no_improvement, int) and self.stop_if_no_improvement > 0
        self._hyperparameters = configs["hyperparameters"]
        assert isinstance(self._hyperparameters, dict)

    @property
    def task(self) -> str:
        """Return the task name."""
        return self._task

    @property
    def mrc_mode(self) -> bool:
        """Return MRC mode status."""
        return self._mrc_mode

    @property
    def dataset_pp(self) -> float:
        """Return the proportion percent of the training set with range (0.0, 1.0]."""
        return self._dataset_pp

    @property
    def data_folder(self) -> str:
        """Return the data folder."""
        return self._data_folder

    @property
    def dataset(self) -> str:
        """Return the dataset name."""
        return self._dataset

    @property
    def datahub_folder(self) -> str:
        """Return the datahub folder."""
        return os.path.join(self.data_folder, "datahub")

    @property
    def dataset_folder(self) -> str:
        """Return the dataset folder."""
        dataset_folder = os.path.join(self.datahub_folder, self.dataset)
        datahub = gctx.create_datahub(self.dataset, root_folder=dataset_folder)
        datahub.preprocess()
        return dataset_folder

    @property
    def modelhub_folder(self) -> str:
        """Return the modelhub folder."""
        return os.path.join(self.data_folder, "modelhub")

    @property
    def tokenizer_model(self) -> str:
        """Return the tokenizer model name."""
        return self._tokenizer_model

    @property
    def tokenizer_model_folder(self) -> str:
        """Return the tokenizer model folder."""
        model_folder = os.path.join(self.modelhub_folder, self._tokenizer_model)
        modelhub = gctx.create_tokenizer(self._tokenizer_model, root_folder=model_folder)
        modelhub.preprocess()
        return model_folder

    @property
    def gazetteer_model_folder(self) -> str:
        """Return the gazetteer model folder."""
        model_folder = os.path.join(self.modelhub_folder, self._gazetteer_model)
        modelhub = gctx.create_gazetteer(self._gazetteer_model, root_folder=model_folder)
        modelhub.preprocess()
        return model_folder

    @property
    def gazetteer_model(self) -> str:
        """Return the gazetteer model name."""
        return self._gazetteer_model

    @property
    def output_folder(self) -> str:
        """Return the output folder."""
        if not os.path.isdir(self._output_folder):
            os.makedirs(self._output_folder)
        return self._output_folder

    @property
    def identity(self) -> str:
        """Return the identity."""
        return self._identity

    @property
    def model(self) -> str:
        """Return the model name."""
        return self._model

    @property
    def in_adapter(self) -> str:
        """Return the in_adapter name."""
        return self._in_adapter

    @property
    def out_adapter(self) -> str:
        """Return the in_adapter name."""
        return self._out_adapter

    @property
    def device(self) -> str:
        """Return the device name."""
        return self._device

    @property
    def gpu(self) -> List[int]:
        """Return the GPU list."""
        return self._gpu

    @property
    def max_seq_len(self) -> int:
        """Return the maximum sequence length."""
        return self._max_seq_len

    @property
    def epoch(self) -> int:
        """Return the epoch number."""
        return self._epoch

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self._batch_size

    @property
    def stop_if_no_improvement(self) -> int:
        """Return stop_if_no_improvement."""
        return self._stop_if_no_improvement

    @property
    def early_stop_loss(self) -> float:
        """Return early_stop_loss."""
        return self._early_stop_loss

    @property
    def optim(self) -> Dict[str, Any]:
        """Return all options of optimization."""
        return self._optim

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, new_hyperparameters: Dict[str, Any]) -> None:
        """Set new hyperparameters."""
        for key, value in new_hyperparameters.items():
            self._hyperparameters[key] = value

    def _load_devices(self) -> None:
        """Initialize devices (gpu, cpu...)."""
        if torch.cuda.is_available() and len(self.gpu) > 0:
            if torch.cuda.device_count() < len(self.gpu):
                self._gpu = list(range(torch.cuda.device_count()))
            self._device = "cuda:{0}".format(self.gpu[0])
        else:
            self._gpu = []

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as a dictionary."""
        configs = {}
        for key, value in vars(self).items():
            if key[0] == "_":
                key = key[1:]
            configs[key] = value

        return configs
