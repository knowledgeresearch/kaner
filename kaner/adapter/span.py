# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Span Schema"""

from typing import List, Tuple


class Span:
    """
    Span is a sub-sequence on a full sequence, which defines the span type
    and its position in the original sequence.

    Args:
        start (int): The start position of the span.
        end (int): The end position of the span.
        label (str): The span type.
        text (text): Span text.
        confidence (float): The probability of the span. If it is a ground truth, then the default
            probability is one.
    """
    def __init__(self, label: str, start: int, end: int, text: str, confidence: float):
        super(Span, self).__init__()
        self.start = start
        self.end = end
        self.label = label
        self.text = text
        self.confidence = confidence

    def __str__(self):
        return "[{0},{1}]{2}".format(self.start, self.end, self.label)

    def __eq__(self, other):
        return str(self) == str(other)


def eliminate_overlap(spans: List[Span]) -> List[Span]:
    """
    Eliminate overlaping spans by the following strategies:
        1) If a recognized entity contains another candidate (nested) entity, only the outer entity
            will be remained for the further processing.
        2) If two identified entities overlap each other, only the one with higher probability is kept.

    Args:
        spans (List[Span]): A list of span.
    """
    spans = sorted(spans, key=lambda span: span.start)
    stack = []
    for span in spans:
        if len(stack) == 0 or span.start > stack[len(stack) - 1].end:
            stack.append(span)
        else:
            top_span = stack[len(stack) - 1]
            if span.end <= top_span.end:
                continue
            elif top_span.end <= span.end and top_span.start >= span.start:
                stack[len(stack) - 1] = span
            elif top_span.confidence < span.confidence:
                stack[len(stack) - 1] = top_span
            else:
                print("func 'eliminate_overlap': same probability", top_span, span)
    return stack


def to_tags(seq_len: int, spans: List[Span]) -> list:
    """
    Convert spans to a list of tag for sequence labeling.
        Tag scheme: BIO.

    Args:
        seq_len (int): The length of sequence.
        spans (List[Span]): All spans on this sequence.
    """
    tags = ["O"] * seq_len
    for span in spans:
        pos = span.start
        if pos < seq_len:
            tags[pos] = "B-{0}".format(span.label)
            pos += 1
        while pos < min(span.end + 1, seq_len):
            tags[pos] = "I-{0}".format(span.label)
            pos += 1

    return tags


def to_spans(tags: List[str], tokens: List[str], probs: List[float]) -> List[Span]:
    """
    Convert a list of tag to spans.
        Tag scheme: BIO.

    Args:
        tags (List[str]): A list of tag.
        tokens (List[str]): A list of tokens.
    """
    assert len(tags) == len(tokens) == len(probs)
    spans = []
    idx = 0
    while idx < len(tags):
        if tags[idx].startswith("B-"):
            label = tags[idx].split("-")[1]
            span = Span(label, idx, idx, tokens[idx], probs[idx])
            idx += 1
            while idx < len(tags) and tags[idx].startswith("I-"):
                span.end += 1
                span.text += tokens[idx]
                span.confidence += probs[idx]
                idx += 1
            span.confidence /= (span.end - span.start + 1)
            spans.append(span)
        else:
            idx += 1

    return spans


def split_document(document: str, spans: List[Span], max_seq_len: int = 512, segsym: str = "。") -> Tuple[List[dict], List[dict]]:
    """
    Split long document into sentences with parameter `max_seq_len` as sliding window.

    Args:
        document (str): The raw document.
        spans (List[Span]): Spans on this document. We do not split document inside a span.
        max_seq_len (str): The maximum sequence length of splited sentences.
        segsym (str): The segmentation symbols.
    """
    if document == "":
        return ([], [])
    # this step should be the first, because the following step would change the document length.
    tags = to_tags(len(document), spans)
    if document.endswith(segsym):
        document = document[:len(document)-1]
        sentences = [sentence + segsym for sentence in document.split(segsym)]
    else:
        sentences = [sentence + segsym for sentence in document.split(segsym)]
        sentences[-1] = sentences[-1][:len(sentences[-1])-1]  # remove last segsym
    data, ignored = [], []
    start = 0
    short_document = ""
    for sentence in sentences:
        if len(short_document) + len(sentence) <= max_seq_len:
            short_document += sentence
        else:
            if len(short_document) > 0:
                data.append({
                    "text": short_document,
                    "spans": to_spans(tags[start: start + len(short_document)], list(short_document), [1.0] * len(short_document))
                })
                start += len(short_document)
                short_document = ""
            if len(sentence) <= max_seq_len:
                short_document = sentence
            else:
                # ignored long fragment
                ignored.append({
                    "text": sentence,
                    "spans": to_spans(tags[start: start + len(sentence)], list(sentence), [1.0] * len(sentence))
                })
                start += len(sentence)
    if short_document != "":
        if len(short_document) <= max_seq_len:
            data.append({
                "text": short_document,
                "spans": to_spans(tags[start: start + len(short_document)], list(short_document), [1.0] * len(short_document))
            })
        else:
            ignored.append({
                "text": sentence,
                "spans": to_spans(tags[start: start + len(sentence)], list(sentence), [1.0] * len(sentence))
            })

    return (data, ignored)
