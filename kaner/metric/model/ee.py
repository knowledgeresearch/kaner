# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Metrics: Event Extraction"""

from typing import Tuple

from kaner.adapter.span import to_spans
from kaner.metric.utils import safe_division


def compute_ee_prf1(pred_batch_tags: list, gold_batch_tags: list) -> Tuple[float, float, float]:
    """
    Compute precision, recall, f1 for named entity recognition.

    TODO: We temporarly consider one event for a given event type only for simplifing preocessing.

    给定一个文档，假设答案包含两个事件实例:

        A1: 事件类型1，要素一取值1，要素二取值2，要素三取值3，要素四取值4
        A2: 事件类型1，要素一取值1，要素二取值6，要素三取值7，要素四取值8

        假设选手给出两个预测结果：
        P1: 事件类型1，要素一取值1，要素二取值2，要素三取值3，要素四取值8
        P2: 事件类型1，要素一取值1，要素二取值2
        P3: 事件类型2，要素一取值9，要素二取值10

    评测时，会采用不放回的方式给每条答案寻找最相似的预测（即事件类型相同且相同要素最多的），例如上例中 A1 与 P1 事件类型相同且 3 个要素相
    同，A1 与 P2 事件类型相同且 2 个要素相同，故与 A1 最相似的预测是 P1，命中 3 个要素。由于采用不放回的方式，此时预测集合剩下 P2、P3，
    与 A2 最相似的预测是 P2，命中 1 个要素。此时两条答案均已找到最相似的预测，可以计算 Precision 和 Recall，如下:


        Precision = (3+1)/(4+2+2)， Recall = (3+1)/(4+4)


        注：在给每条答案寻找最相似预测时，相同事件类型的答案会按照要素个数的多少定优先级，如上例中 A1 和 A2 事件类型相同，但 A1 要素个数
        多，故优先为 A1 寻找最相似预测。

        事件要素精确率=识别事件类型与要素和标注相同的数量/识别出事件类型与要素总数量

        事件要素召回率=识别出事件类型与要素和标注相同的数量/标注的事件类型与要素总数量

        事件要素F1值=(2事件要素精确率事件要素召回率)/(事件要素精确率+事件要素召回率)

    Args:
        pred_batch_tags (list): A batch of sentence tags from predictions.
        gold_batch_tags (list): A batch of sentence tags from gold annotations.
    """
    num_sentences = len(pred_batch_tags)
    true_pred, pred_samples, gold_samples = [], [], []
    for i in range(num_sentences):
        for event_id, sequences in enumerate(pred_batch_tags[i]):
            pred_arguments = to_spans(pred_batch_tags[i][event_id], ["X"]*len(pred_batch_tags[i][event_id]), [1.0]*len(pred_batch_tags[i][event_id]))
            gold_arguments = to_spans(gold_batch_tags[i][event_id], ["X"]*len(gold_batch_tags[i][event_id]), [1.0]*len(gold_batch_tags[i][event_id]))
            pred = [str(argument) for argument in pred_arguments]
            gold = [str(argument) for argument in gold_arguments]
            pred_samples.extend(pred)
            gold_samples.extend(gold)
            true_pred.extend(list(set(pred).intersection(set(gold))))
    precision = safe_division(len(true_pred), len(pred_samples))
    recall = safe_division(len(true_pred), len(gold_samples))
    f1 = safe_division(2*precision*recall, precision+recall)

    return (precision, recall, f1)
