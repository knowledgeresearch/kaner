# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Input Adapter for Sequence Labeling + MDGG"""

from typing import Dict, Any, Union, List, Tuple
from copy import deepcopy

from kaner.context import GlobalContext as gctx
from kaner.adapter.tokenizer import BaseTokenizer
from kaner.adapter.out_adapter import BaseOutAdapter
from kaner.adapter.knowledge import Gazetteer
from kaner.adapter.span import to_tags, Span
from .base import BaseInAdapter


@gctx.register_inadapter("in_mdgg")
class InMDGG(BaseInAdapter):
    """
    InMDGG is a sub-class of BaseInAdapter for the task "Sequence Labeling" + "MDGG".

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
        super(InMDGG, self).__init__(dataset, max_seq_len, tokenizer, out_adapter, _gazetteer=gazetteer, _queries=queries)
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

        if self._queries is None:
            tokens = self._tokenizer.tokenize(text)[:self._max_seq_len]
            labels = to_tags(len(tokens), spans)[:self._max_seq_len]
            lexicons, relations = self._build_graph(tokens, 0)
            token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            label_ids = self._out_adapter.convert_labels_to_ids(labels)
            lexicon_ids = [self._gazetteer[lexicon] for lexicon in lexicons]
            datum = {
                "text": text,
                "input_ids": token_ids,
                "output_ids": label_ids,
                "lexicon_ids": lexicon_ids,
                "relations": relations,
                "length": len(token_ids),
                "start": 0
            }

            return datum

        datum_group = []
        for span_type, query in self._queries.items():
            q_tokens = [self._cls_token] + list(query) + [self._sep_token] + self._tokenizer.tokenize(text)
            q_tokens = q_tokens[:self._max_seq_len]
            q_token_ids = self._tokenizer.convert_tokens_to_ids(q_tokens)
            q_spans = deepcopy(list(filter(lambda span: span.label == span_type, spans)))
            for i, _ in enumerate(q_spans):
                q_spans[i].label = "Span"
            q_labels = [self._out_adapter.unk_label]*(2 + len(query)) + to_tags(len(text), q_spans)
            q_labels = q_labels[:self._max_seq_len]
            q_label_ids = self._out_adapter.convert_labels_to_ids(q_labels)
            q_lexicons, q_relations = self._build_graph(q_tokens, 2 + len(query))
            q_lexicon_ids = [self._gazetteer[lexicon] for lexicon in q_lexicons]
            datum_group.append(
                {
                    "type": span_type,
                    "query": query,
                    "text": text,
                    "input_ids": q_token_ids,
                    "output_ids": q_label_ids,
                    "lexicon_ids": q_lexicon_ids,
                    "relations": q_relations,
                    "length": len(q_token_ids),
                    "start": min(2 + len(query), len(q_token_ids))
                }
            )

        return datum_group

    def _build_graph(self, tokens: List[str], start: int = 0) -> Tuple[list, list]:
        """
        Build the adjcent matrix of three kinds of graph according to edge types.

        Args:
            tokens (List[str]): A list of input tokens.
            start (int): The start point of lexicon matching.
        """
        lexicons = []
        # relation: (node_id, is_lexicon)
        relations = [[] for _ in range(self._gazetteer.num_types + 1)]
        reltype2id = {"CHARACTER": 0}
        # connect characters
        edge_type_id = reltype2id["CHARACTER"]
        for i in range(1, len(tokens)):
            relations[edge_type_id].append(((i - 1, False), (i, False)))
        for i in range(start, len(tokens)):
            matched_lexicons = self._gazetteer.search(tokens[i:])
            for lexicon in matched_lexicons:
                lexicon_type = self._gazetteer.lexicon_type(lexicon)
                if lexicon_type not in reltype2id.keys():
                    reltype2id[lexicon_type] = len(reltype2id)
                # connect lexicons: start -> first-char -> ... -> last-char -> end
                edge_type_id = reltype2id[lexicon_type]
                lexicons.extend([lexicon, lexicon])
                start_node, end_node = len(lexicons) - 2, len(lexicons) - 1
                relations[edge_type_id].append(((start_node, True), (i, False)))
                relations[edge_type_id].append(((i + len(lexicon) - 1, False), (end_node, True)))
                for i in range(i, i + len(lexicon) - 1):
                    relations[edge_type_id].append(((i, False), (i + 1, False)))

        return lexicons, relations
