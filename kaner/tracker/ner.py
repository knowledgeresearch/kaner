# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""NERTracker"""

import os
from typing import Dict, Any, List
import pandas as pd
from .base import BaseTrackerRow, BaseTracker


class NERTrackerRow(BaseTrackerRow):
    """
    Track key metrics of an experiment in a NER task.

    Args:
        date (str): The start date of running a task.
        model (str): The model name.
        dataset (str): The dataset name.
        tokenizer_model (str): Tokenizer model name.
        gazetteer_model str): Gazetteer model name.
        log_dir (str): Log folder.
        time_consu (float): The total time cost of running a task (SEC).
        f1_score (float): F1 score.
        precision_score (float): Precision score.
        recall_score (float): Recall score.
        total_epoch (int): The total number of epochs for training the model.
        test_loss (float): Loss value in the test set.
        tag (str): Experimental tags.
    """
    __default_values__ = {
        "date": "2020-01-01",
        "model": "nil", "dataset": "nil", "tokenizer_model": "nil", "gazetteer_model": "nil",
        "log_dir": "nil",
        "f1_score": 0., "precision_score": 0., "recall_score": 0.,
        "time_consu": 0., "total_epoch": 0, "test_loss": 0., "time_per_epoch": 0.,
        "tag": "nil"
    }

    def __init__(
            self,
            date: str,
            model: str,
            dataset: str,
            tokenizer_model: str,
            gazetteer_model: str,
            log_dir: str,
            time_consu: float,
            f1_score: float,
            precision_score: float,
            recall_score: float,
            total_epoch: int,
            test_loss: float,
            tag: str
    ):
        super(NERTrackerRow, self).__init__()
        self._date = date
        self._model = model
        self._dataset = dataset
        self._tokenizer_model = tokenizer_model
        self._gazetteer_model = gazetteer_model
        self._log_dir = log_dir
        self._time_consu = time_consu
        self._f1_score = f1_score
        self._precision_score = precision_score
        self._recall_score = recall_score
        self._total_epoch = total_epoch
        self._test_loss = test_loss
        self._time_per_epoch = time_consu / (total_epoch + 1e-6)
        self._tag = tag

    @property
    def date(self) -> str:
        """
        The start date of running a task.
        """
        return self._date

    @property
    def model(self) -> str:
        """
        The model name.
        """
        return self._model

    @property
    def dataset(self) -> str:
        """
        The dataset name.
        """
        return self._dataset

    @property
    def tokenizer_model(self) -> str:
        """
        The tokenizer model name.
        """
        return self._tokenizer_model

    @property
    def gazetteer_model(self) -> str:
        """
        The gazetteer model name.
        """
        return self._gazetteer_model

    @property
    def log_dir(self) -> str:
        """
        The log directory.
        """
        return self._log_dir

    @property
    def f1_score(self) -> float:
        """
        F1 score.
        """
        return self._f1_score

    @property
    def precision_score(self) -> float:
        """
        Precision score.
        """
        return self._precision_score

    @property
    def recall_score(self) -> float:
        """
        Recall score.
        """
        return self._recall_score

    @property
    def time_consu(self) -> float:
        """
         The total time cost of running a task (SEC).
        """
        return self._time_consu

    @property
    def total_epoch(self) -> int:
        """
        The total number of epochs for training the model.
        """
        return self._total_epoch

    @property
    def test_loss(self) -> float:
        """Return the loss in the test set"""
        return self._test_loss

    @property
    def time_per_epoch(self) -> float:
        """
        Time cost per epoch.
        """
        return self._time_per_epoch

    @property
    def tag(self) -> str:
        """
        Experimental tags.
        """
        return self._tag


class NERTracker(BaseTracker):
    """
    NERTracker for tracking the key metrics of the NER task.

    Args:
        ner_tracker_row (NERTrackerRow): A class of NERTrackerRow.
    """

    def __init__(self, ner_tracker_row: NERTrackerRow):
        super(NERTracker, self).__init__(ner_tracker_row)

    @classmethod
    def load(cls, labpath: str) -> BaseTracker:
        """
        Load table from csv file.

        Args:
            labpath (str): The file path of recording experimental results.
        """
        tracker = cls(NERTrackerRow)
        folder = os.path.dirname(labpath)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if os.path.isfile(labpath):
            data = pd.read_csv(labpath)
            cols1 = list(data.columns)
            cols2 = list(tracker._table.columns)
            for col in cols2:
                if col not in cols1:
                    data[col] = NERTrackerRow.__default_values__[col]
            for row in data.to_dict("records"):
                row.pop("time_per_epoch")
                trakcer_row = NERTrackerRow(**row)
                tracker.insert(trakcer_row)

        return tracker

    def summay(self) -> List[Dict[str, Any]]:
        """
        Calculate summary statistics according to a given condition.
        """
        models = list(set(self._table["model"].to_list()))
        datasets = list(set(self._table["dataset"].to_list()))
        tokenizer_models = list(set(self._table["tokenizer_model"].to_list()))
        gazetteer_models = list(set(self._table["gazetteer_model"].to_list()))
        results = []
        for model in models:
            for dataset in datasets:
                for tokenizer_model in tokenizer_models:
                    for gazetteer_model in gazetteer_models:
                        records = self.query(
                            model=model, dataset=dataset, tokenizer_model=tokenizer_model, gazetteer_model=gazetteer_model
                        )
                        if len(records) == 0:
                            continue
                        tags = list(set(records["tag"].to_list()))
                        for tag in tags:
                            sub_records = records.query("{0} == '{1}'".format("tag", tag))
                            results.append({
                                "model": model,
                                "dataset": dataset,
                                "tokenizer_model": tokenizer_model,
                                "gazetteer_model": gazetteer_model,
                                "n_experiments": len(sub_records),
                                "tag": tag,
                                "f1_score": sub_records["f1_score"].mean(),
                                "precision_score": sub_records["precision_score"].mean(),
                                "recall_score": sub_records["recall_score"].mean(),
                                "time_consu": sub_records["time_consu"].mean(),
                                "total_epoch": sub_records["total_epoch"].mean(),
                                "test_loss": sub_records["test_loss"].mean(),
                                "time_per_epoch": sub_records["time_per_epoch"].mean()
                            })

        return results
