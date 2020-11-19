# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Batch Utils
    Batcher mainly contains collate_fn which is called with a list of data samples at each time.
    It is expected to collate the input samples into a batch for yielding from the data loader
    iterator.
"""

from typing import Callable, Tuple, List

import torch

from kaner.context import GlobalContext as gctx


def batch_seq_mask(lengths: List[int]) -> List[List[bool]]:
    """
    Return batch sequence mask.

    Args:
        lengths (List[int]): A list of batch sequence length.
    """
    max_seq_len = max(lengths)
    mask = []
    for length in lengths:
        mask.append([False] * length + [True] * (max_seq_len - length))

    return mask


@gctx.register_batcher("blcrf")
def cfn_blcrf(input_pad: int, output_pad: int, device: str) -> Callable:
    """
    Collate function for the model BLCRF.

    Args:
        input_pad (int): Align all inputs by PAD.
        output_pad (int): Align all outputs by PAD.
        device (str): Device used to compute tensors. The commonly used value is cpu or cuda.
    """
    def collate_fn(batch) -> Tuple[tuple, dict]:
        max_seq_len = 1
        for i, _ in enumerate(batch):
            max_seq_len = max(max_seq_len, len(batch[i]["input_ids"]))
        batch_input_ids, batch_output_ids = [], []
        batch_lengths = []
        for i, _ in enumerate(batch):
            input_ids = batch[i]["input_ids"] + [input_pad] * (max_seq_len - len(batch[i]["input_ids"]))
            batch_input_ids.append(input_ids)
            output_ids = batch[i]["output_ids"] + [output_pad] * (max_seq_len - len(batch[i]["output_ids"]))
            batch_output_ids.append(output_ids)
            batch_lengths.append(batch[i]["length"])
        ready_to_eat = {
            "inputs": torch.tensor(batch_input_ids).to(device),
            "outputs": torch.tensor(batch_output_ids).to(device),
            "lengths": torch.tensor(batch_lengths).to(device)
        }

        return (batch, ready_to_eat)

    return collate_fn


@gctx.register_batcher("plmtg")
def cfn_plmtg(input_pad: int, output_pad: int, device: str) -> Callable:
    """
    Collate function for the model PLMTG.

    Args:
        input_pad (int): Align all inputs by PAD.
        output_pad (int): Align all outputs by PAD.
        device (str): Device used to compute tensors. The commonly used value is cpu or cuda.
    """
    def collate_fn(batch) -> Tuple[tuple, dict]:
        max_seq_len = 1
        for i, _ in enumerate(batch):
            max_seq_len = max(max_seq_len, len(batch[i]["input_ids"]))
        batch_input_ids, batch_output_ids = [], []
        batch_lengths = []
        for i, _ in enumerate(batch):
            input_ids = batch[i]["input_ids"] + [input_pad] * (max_seq_len - len(batch[i]["input_ids"]))
            batch_input_ids.append(input_ids)
            output_ids = batch[i]["output_ids"] + [output_pad] * (max_seq_len - len(batch[i]["output_ids"]))
            batch_output_ids.append(output_ids)
            batch_lengths.append(batch[i]["length"])
        ready_to_eat = {
            "inputs": torch.tensor(batch_input_ids).to(device),
            "outputs": torch.tensor(batch_output_ids).to(device),
            "lengths": torch.tensor(batch_lengths).to(device)
        }

        return (batch, ready_to_eat)

    return collate_fn


@gctx.register_batcher("ses")
def cfn_ses(input_pad: int, output_pad: int, lexicon_pad: int, device: str) -> Callable:
    """
    Collate function for the model SES.

    Args:
        input_pad (int): Align all inputs by PAD.
        output_pad (int): Align all outputs by PAD.
        lexicon_pad (int): Algin matched lexicon sets for each position by PAD.
        device (str): Device used to compute tensors. The commonly used value is cpu or cuda.
    """
    def collate_fn(batch) -> Tuple[tuple, dict]:
        max_seq_len, max_n_lexicons = 1, 1
        n_sets = 0
        for i, _ in enumerate(batch):
            max_seq_len = max(max_seq_len, len(batch[i]["input_ids"]))
            for lexicon_sets in batch[i]["lexicon_ids"]:
                # four sets: B M E S
                n_sets = len(lexicon_sets)
                for lexicon_set in lexicon_sets:
                    max_n_lexicons = max(max_n_lexicons, len(lexicon_set))
        batch_input_ids, batch_output_ids = [], []
        batch_lengths = []
        batch_lexicon_ids, batch_weights = [], []
        for i, _ in enumerate(batch):
            input_ids = batch[i]["input_ids"] + [input_pad] * (max_seq_len - len(batch[i]["input_ids"]))
            batch_input_ids.append(input_ids)
            output_ids = batch[i]["output_ids"] + [output_pad] * (max_seq_len - len(batch[i]["output_ids"]))
            batch_output_ids.append(output_ids)
            batch_lengths.append(batch[i]["length"])

            batch_lexicon_ids.append(batch[i]["lexicon_ids"])
            batch_weights.append(batch[i]["weights"])
            for pos, _ in enumerate(batch_lexicon_ids[i]):
                for set_id, _ in enumerate(batch_lexicon_ids[i][pos]):
                    batch_lexicon_ids[i][pos][set_id] += [lexicon_pad] * (max_n_lexicons - len(batch_lexicon_ids[i][pos][set_id]))
                    batch_weights[i][pos][set_id] += [0.0] * (max_n_lexicons - len(batch_weights[i][pos][set_id]))
            batch_lexicon_ids[i] += [
                [[lexicon_pad] * max_n_lexicons for _ in range(n_sets)] for _ in range(max_seq_len - len(batch_lexicon_ids[i]))
            ]
            batch_weights[i] += [
                [[0.0] * max_n_lexicons for _ in range(n_sets)] for _ in range(max_seq_len - len(batch_weights[i]))
            ]
        ready_to_eat = {
            "inputs": torch.tensor(batch_input_ids).to(device),
            "outputs": torch.tensor(batch_output_ids).to(device),
            "lengths": torch.tensor(batch_lengths).to(device),
            "lexicons": torch.tensor(batch_lexicon_ids).to(device),
            "weights": torch.tensor(batch_weights).to(device)
        }

        return (batch, ready_to_eat)

    return collate_fn


@gctx.register_batcher("cgn")
def cfn_cgn(input_pad: int, output_pad: int, lexicon_pad: int, device: str) -> Callable:
    """
    Collate function for the model CGN.

    Args:
        input_pad (int): Align all inputs by PAD.
        output_pad (int): Align all outputs by PAD.
        lexicon_pad (int): Algin matched lexicon sets for each position by PAD.
        device (str): Device used to compute tensors. The commonly used value is cpu or cuda.
    """
    def collate_fn(batch) -> Tuple[tuple, dict]:
        max_seq_len, max_n_lexicons = 1, 1
        for i, _ in enumerate(batch):
            max_seq_len = max(max_seq_len, len(batch[i]["input_ids"]))
            max_n_lexicons = max(max_n_lexicons, len(batch[i]["lexicon_ids"]))
        n_nodes = max_seq_len + max_n_lexicons
        batch_input_ids, batch_output_ids = [], []
        batch_lengths = []
        batch_lexicon_ids, batch_graphs = [], []
        for i, _ in enumerate(batch):
            input_ids = batch[i]["input_ids"] + [input_pad] * (max_seq_len - len(batch[i]["input_ids"]))
            batch_input_ids.append(input_ids)
            output_ids = batch[i]["output_ids"] + [output_pad] * (max_seq_len - len(batch[i]["output_ids"]))
            batch_output_ids.append(output_ids)
            lexicon_ids = batch[i]["lexicon_ids"] + [lexicon_pad] * (max_n_lexicons - len(batch[i]["lexicon_ids"]))
            batch_lexicon_ids.append(lexicon_ids)
            batch_lengths.append(batch[i]["length"])

            graphs = []
            for relations in batch[i]["relations"]:
                graph = [[0] * n_nodes for _ in range(n_nodes)]
                for i in range(n_nodes):
                    graph[i][i] = 1
                for relation in relations:
                    start, end = relation[0][0], relation[1][0]
                    if relation[0][1]:
                        start += max_seq_len
                    if relation[1][1]:
                        end += max_seq_len
                    graph[start][end] = 1
                    graph[end][start] = 1
                graphs.append(graph)
            batch_graphs.append(graphs)
        ready_to_eat = {
            "inputs": torch.tensor(batch_input_ids).to(device),
            "outputs": torch.tensor(batch_output_ids).to(device),
            "lengths": torch.tensor(batch_lengths).to(device),
            "lexicons": torch.tensor(batch_lexicon_ids).to(device),
            "graphs": torch.tensor(batch_graphs).to(device)
        }

        return (batch, ready_to_eat)

    return collate_fn


@gctx.register_batcher("mdgg")
def cfn_mdgg(input_pad: int, output_pad: int, lexicon_pad: int, device: str) -> Callable:
    """
    Collate function for the model MDGG.

    Args:
        input_pad (int): Align all inputs by PAD.
        output_pad (int): Align all outputs by PAD.
        lexicon_pad (int): Algin matched lexicon sets for each position by PAD.
        device (str): Device used to compute tensors. The commonly used value is cpu or cuda.
    """
    def collate_fn(batch) -> Tuple[tuple, dict]:
        max_seq_len, max_n_lexicons = 1, 1
        for i, _ in enumerate(batch):
            max_seq_len = max(max_seq_len, len(batch[i]["input_ids"]))
            max_n_lexicons = max(max_n_lexicons, len(batch[i]["lexicon_ids"]))
        n_nodes = max_seq_len + max_n_lexicons
        batch_input_ids, batch_output_ids = [], []
        batch_lengths = []
        batch_lexicon_ids, batch_graphs = [], []
        for i, _ in enumerate(batch):
            input_ids = batch[i]["input_ids"] + [input_pad] * (max_seq_len - len(batch[i]["input_ids"]))
            batch_input_ids.append(input_ids)
            output_ids = batch[i]["output_ids"] + [output_pad] * (max_seq_len - len(batch[i]["output_ids"]))
            batch_output_ids.append(output_ids)
            lexicon_ids = batch[i]["lexicon_ids"] + [lexicon_pad] * (max_n_lexicons - len(batch[i]["lexicon_ids"]))
            batch_lexicon_ids.append(lexicon_ids)
            batch_lengths.append(batch[i]["length"])

            graphs = []
            for relations in batch[i]["relations"]:
                graph = [[0] * n_nodes for _ in range(n_nodes)]
                for i in range(n_nodes):
                    graph[i][i] = 1
                for relation in relations:
                    start, end = relation[0][0], relation[1][0]
                    if relation[0][1]:
                        start += max_seq_len
                    if relation[1][1]:
                        end += max_seq_len
                    # directed graphs
                    graph[start][end] = 1
                graphs.append(graph)
            batch_graphs.append(graphs)
        ready_to_eat = {
            "inputs": torch.tensor(batch_input_ids).to(device),
            "outputs": torch.tensor(batch_output_ids).to(device),
            "lengths": torch.tensor(batch_lengths).to(device),
            "lexicons": torch.tensor(batch_lexicon_ids).to(device),
            "graphs": torch.tensor(batch_graphs).to(device)
        }

        return (batch, ready_to_eat)

    return collate_fn
