# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Base Input Adapter"""

from typing import Dict, Any, Union, List

from torch.utils.data import Dataset
import tqdm

from kaner.adapter.tokenizer import BaseTokenizer
from kaner.adapter.out_adapter import BaseOutAdapter
from kaner.adapter.out_adapter.mrc import OutMRC


class BaseInAdapter(Dataset):
    """
    BaseInAdapter defines the basic interfaces for the dataset pre-processing.

    Args:
        dataset (list): The dataset with format list.
        max_seq_len (int): The maximum length of an input text.
        tokenizer (BaseTokenizer): Tokenizer for tokenizing the input text.
        out_adapter (BaseOutAdapter): The output adapter for data post-processing.
        additional_attr (dict): Additional attributes, such as gazetteer, etc.
    """

    def __init__(
            self,
            dataset: list,
            max_seq_len: int,
            tokenizer: BaseTokenizer,
            out_adapter: BaseOutAdapter,
            **additional_attr
    ):
        super(BaseInAdapter, self).__init__()
        self.raw = dataset
        self._max_seq_len = max_seq_len
        self._tokenizer = tokenizer
        self._out_adapter = out_adapter
        self._cls_token = "[CLS]"
        self._sep_token = "[SEP]"
        if isinstance(out_adapter, OutMRC):
            assert "_queries" in additional_attr.keys() and additional_attr["_queries"] is not None
            assert all([tokenizer.exist(sptok) for sptok in [self._cls_token, self._sep_token]]),\
                "{0} should in the tokenizer in the mode of MRC".format([self._cls_token, self._sep_token])
            print("# MRC Mode")
        for key, obj in additional_attr.items():
            setattr(self, key, obj)
        self._data = []
        self._transform_all()

    def transform_sample(self, sample: Dict[str, Any]) -> Union[Dict[str, Any], List[dict]]:
        """
        In order to feed the data into neural networks, we transform sample to numbers.

        Args:
            sample (sample: Dict[str, Any]): Sample to be converted.
        """
        raise NotImplementedError

    def _transform_all(self) -> None:
        """
        In order to feed the data into neural networks, we transform all samples to numbers.
        """
        if self._queries is not None:
            for sample in self.raw:
                for span in sample["spans"]:
                    assert span["label"] in self._queries.keys(), \
                        "Span label {0} should in query keys {1}".format(span["label"], list(self._queries.keys()))
        for sample in tqdm.tqdm(self.raw, "Text2Tensor"):
            datum = self.transform_sample(sample)
            if isinstance(datum, dict):
                datum = [datum]
            assert isinstance(datum, list)
            self._data.extend(datum)

    def __len__(self) -> int:
        """
        Return the number of samples.
        """
        return len(self._data)

    def __getitem__(self, item) -> dict:
        """
        Return the corresponding sample.
        """
        return self._data[item]
