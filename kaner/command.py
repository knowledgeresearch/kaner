# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Command Line Tool"""

import fire

from kaner.trainer import TrainerConfig
from kaner.pipeline import train
from kaner.service import serve
from kaner.common import set_seed


class CLI:
    """
    Command Line Interfaces of KANER (Knowledge-Aware Named Entity Recognition). KANER is a toolkit with various
    knowledge-enhanced NER models aiming to facilitate development, expriment and test of these models. This tool
    mainly provides data processing, model training, and model inference.
    """

    @staticmethod
    def train(cfgpath: str, **kwargs) -> None:
        """
        When this command is executed, a training experiment with a given configuration will be conducted.\n
        :param cfgpath (str): The configuration path where the configuration file decribes all configurations that
            a trainer needed.
        """
        config = TrainerConfig(cfgpath, **kwargs)
        train(config)

    @staticmethod
    def serve(mfolder: str, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        When this command is executed, it will start a service for predicting texts.\n
        :param mfolder (str): The folder where the trained model exists.\n
        :param host (str): Network address.\n
        :param port (str): Listen port.
        """
        serve(mfolder, host, port)


def command() -> None:
    """Start CLI."""
    set_seed(9999)
    fire.Fire(CLI)
