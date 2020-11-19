# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""BaseTracker"""

from typing import Any
import pandas as pd


class BaseTrackerRow:
    """
    A row records for Tracker to check values.
    """
    __default_values__ = {}

    def __init__(self):
        super(BaseTrackerRow, self).__init__()


class BaseTracker:
    """
    BaseTracker provides some general utils for tracking model's metrics. Besides, it provides some interfaces
    to thoese subclasses to implement, such as load, and save.

    Args:
        tracker_row (BaseTrackerRow): A class of BaseTrackerRow.
    """

    def __init__(self, tracker_row):
        super(BaseTracker, self).__init__()
        self._tracker_row = tracker_row
        cols = [attr for attr in tracker_row.__dict__.keys() if not attr.startswith("_")]
        assert set(cols) == set(tracker_row.__default_values__.keys())
        self._table = pd.DataFrame(columns=cols)

    def insert(self, row: BaseTrackerRow) -> None:
        """
        Insert a row in the table.

        Args:
            row (BaseTrackerRow): A new raw.
        """
        assert isinstance(row, self._tracker_row)
        row = [row.__dict__["_" + col] for col in list(self._table.columns)]
        self._table.loc[len(self._table)] = row

    def save(self, labpath: str) -> None:
        """
        Save table with format csv.

        Args:
            labpath (str): The file path of recording experimental results.
        """
        self._table.to_csv(labpath, index=False)
        print("# Save experimental data into {0}".format(labpath))

    @classmethod
    def load(cls, labpath: str) -> None:
        """
        Load table from csv file.

        Args:
            labpath (str): The file path of recording experimental results.
        """
        raise NotImplementedError

    def query(self, **columns) -> pd.DataFrame:
        """
        Query rows according to colname-value pair.

        Args:
            columns (dict): Column_name-value pair.

        Example:
            name | address
             --  |  ---
             n1  |  addr1
             n2  |  addr2
        >>> columns = {"name": "n1", "address": "addr1"}
        """
        conditions = ["{0} == '{1}'".format(col, val) for col, val in columns.items()]
        results = self._table.query(" & ".join(conditions))

        return results

    def summay(self, *args, **kwargs) -> Any:
        """
        Calculate summary statistics according to a given condition.
        """
        raise NotImplementedError
