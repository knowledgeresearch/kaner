# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Input Adapter for Sequence Labeling + SES"""

from typing import Dict, Any, Union, List
from copy import deepcopy

from kaner.context import GlobalContext as gctx
from kaner.adapter.tokenizer import BaseTokenizer
from kaner.adapter.out_adapter import BaseOutAdapter
from kaner.adapter.knowledge import Gazetteer
from kaner.adapter.span import to_tags, Span

from .base import BaseInAdapter


@gctx.register_inadapter("in_ses")
class InSES(BaseInAdapter):
    """
    InSES is a sub-class of BaseInAdapter for the task "Sequence Labeling" + "SES".

    Args:
        gazetteer (Gazetteer): Gazetteer adapter.
        queries (Dict[str, str]): Queries for each entity type in the mode of MRC.
    """

    def __init__(
            self,
            dataset: list,
            max_seq_len: int,
            tokenizer: BaseTokenizer,
            out_adapter: BaseOutAdapter,
            gazetteer: Gazetteer,
            queries: Dict[str, str] = None
    ):
        super(InSES, self).__init__(
            dataset, max_seq_len, tokenizer, out_adapter, _gazetteer=gazetteer, _queries=queries
        )
        # The following one line codes are not necessary. Just for
        # stopping reportting error by pylint.
        self._gazetteer = gazetteer
        self._queries = queries

    def transform_sample(self, sample: Dict[str, Any]) -> Union[Dict[str, Any], List[dict]]:
        """
        In order to feed the data into neural networks, we transform sample to numbers.

        Args:
            sample (sample: Dict[str, Any]): Sample to be converted.
        """
        text = sample["text"]
        spans = [Span(span["label"], span["start"], span["end"], sample["text"][span["start"]: span["end"] + 1], 1.0) for span in sample["spans"]]

        tokens = self._tokenizer.tokenize(text)
        labels = to_tags(len(tokens), spans)
        # sets: B=[], M=[], E=[], S=[]
        lexicon_sets = [[[], [], [], []] for _, _ in enumerate(tokens)]
        for i, token in enumerate(tokens):
            matched_lexicons = self._gazetteer.search(tokens[i:])
            for lexicon in matched_lexicons:
                if len(lexicon) == 1:
                    lexicon_sets[i][3].append(lexicon)  # S
                else:
                    lexicon_sets[i][0].append(lexicon)  # B
                    lexicon_sets[i + len(lexicon) - 1][2].append(lexicon)   # E
                    for j in range(1, len(lexicon) - 1):
                        lexicon_sets[i + j][1].append(lexicon)  # M
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        label_ids = self._out_adapter.convert_labels_to_ids(labels)
        # (seq_len, n_sets, n_lexicons)
        weights = []
        lexicon_ids = []
        for i in range(len(lexicon_sets)):
            weights.append([[] for _ in range(len(lexicon_sets[i]))])
            lexicon_ids.append([[] for _ in range(len(lexicon_sets[i]))])
            for j in range(len(lexicon_sets[i])):
                for k in range(len(lexicon_sets[i][j])):
                    lexicon_ids[i][j].append(self._gazetteer[lexicon_sets[i][j][k]])
                    # NOTE: we use average pooling, because we find weighted pooling has negative effects
                    #       against the results from the original paper.
                    weights[i][j].append(1.0 / len(lexicon_sets[i][j]))

        if self._queries is None:
            datum = {
                "text": text,
                "input_ids": token_ids[:self._max_seq_len],
                "output_ids": label_ids[:self._max_seq_len],
                "lexicon_ids": lexicon_ids[:self._max_seq_len],
                "weights": weights[:self._max_seq_len],
                "length": len(token_ids[:self._max_seq_len]),
                "start": 0
            }

            return datum

        datum_group = []
        for span_type, query in self._queries.items():
            query_ids = self._tokenizer.convert_tokens_to_ids([self._cls_token] + list(query) + [self._sep_token])
            q_token_ids = query_ids + deepcopy(token_ids)
            q_spans = deepcopy(list(filter(lambda span: span.label == span_type, spans)))
            for i, _ in enumerate(q_spans):
                q_spans[i].label = "Span"
            q_labels = [self._out_adapter.unk_label]*(2 + len(query)) + to_tags(len(text), q_spans)
            q_label_ids = self._out_adapter.convert_labels_to_ids(q_labels)
            q_lexicon_ids = [[[], [], [], []] for _ in range(2 + len(query))] + deepcopy(lexicon_ids)
            q_weights = [[[], [], [], []] for _ in range(2 + len(query))] + deepcopy(weights)
            datum_group.append(
                {
                    "type": span_type,
                    "query": query,
                    "text": text,
                    "input_ids": q_token_ids[:self._max_seq_len],
                    "output_ids": q_label_ids[:self._max_seq_len],
                    "lexicon_ids": q_lexicon_ids[:self._max_seq_len],
                    "weights": q_weights[:self._max_seq_len],
                    "length": len(q_token_ids[:self._max_seq_len]),
                    "start": min(2 + len(query), len(q_token_ids[:self._max_seq_len]))
                }
            )

        return datum_group
