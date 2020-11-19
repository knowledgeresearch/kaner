# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Input Adapter for Sequence Labeling"""

from typing import Dict, Any, Union, List
from copy import deepcopy

from kaner.context import GlobalContext as gctx
from kaner.adapter.tokenizer import BaseTokenizer
from kaner.adapter.out_adapter import BaseOutAdapter
from kaner.adapter.span import to_tags, Span
from .base import BaseInAdapter


@gctx.register_inadapter("in_seqlab")
class InSeqlab(BaseInAdapter):
    """
    InSeqlab is a sub-class of BaseInAdapter for the task "Sequence Labeling".

    Args:
        queries (Dict[str, str]): Queries for each entity type in the mode of MRC.
    """

    def __init__(
            self,
            dataset: list,
            max_seq_len: int,
            tokenizer: BaseTokenizer,
            out_adapter: BaseOutAdapter,
            queries: Dict[str, str] = None
    ):
        super(InSeqlab, self).__init__(
            dataset, max_seq_len, tokenizer, out_adapter, _queries=queries
        )
        # The following one line codes are not necessary. Just for
        # stopping reportting error by pylint.
        self._queries = queries

    def transform_sample(self, sample: Dict[str, Any]) -> Union[Dict[str, Any], List[dict]]:
        """
        In order to feed the data into neural networks, we transform sample to numbers.

        Args:
            sample (sample: Dict[str, Any]): Sample to be converted.
        """
        text = sample["text"]
        spans = [Span(span["label"], span["start"], span["end"], sample["text"][span["start"]: span["end"] + 1], 1.0) for span in sample["spans"]]
        if self._queries is None:
            tokens = self._tokenizer.tokenize(text)
            token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            labels = to_tags(len(tokens), spans)
            label_ids = self._out_adapter.convert_labels_to_ids(labels)
            datum = {
                "text": text,
                "input_ids": token_ids[:self._max_seq_len],
                "output_ids": label_ids[:self._max_seq_len],
                "length": len(token_ids[:self._max_seq_len]),
                "start": 0
            }

            return datum

        datum_group = []
        for span_type, query in self._queries.items():
            q_tokens = [self._cls_token] + list(query) + [self._sep_token] + list(text)
            q_token_ids = self._tokenizer.convert_tokens_to_ids(q_tokens)
            q_spans = deepcopy(list(filter(lambda span: span.label == span_type, spans)))
            for i, _ in enumerate(q_spans):
                q_spans[i].label = "Span"
            q_labels = [self._out_adapter.unk_label]*(2 + len(query)) + to_tags(len(text), q_spans)
            q_label_ids = self._out_adapter.convert_labels_to_ids(q_labels)
            datum_group.append(
                {
                    "type": span_type,
                    "query": query,
                    "text": text,
                    "input_ids": q_token_ids[:self._max_seq_len],
                    "output_ids": q_label_ids[:self._max_seq_len],
                    "length": len(q_token_ids[:self._max_seq_len]),
                    "start": min(2 + len(query), len(q_token_ids[:self._max_seq_len]))
                }
            )

        return datum_group
