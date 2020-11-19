# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Event Schema"""

from typing import List
from .span import Span


class Event:
    """
    In philosophy, events are objects in time or instantiations of properties in objects. It contains event type
    and several corresponding arguments to describing the event. For example, an Asset Loss event may contains
    arguments such as company name and date.

    Args:
        event_type (str): The event type.
        confidence (float): The probability of the event.
        arguments (List[Span]): All event arguments for supporting this event.
    """
    def __init__(self, event_type: str, confidence: float, arguments: List[Span]):
        super(Event, self).__init__()
        self.event_type = event_type
        self.confidence = confidence
        self.arguments = arguments

    def __str__(self):
        return "#{0}#{1}".format(self.event_type, ";".join([str(argument) for argument in self.arguments]))

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def __dict__(self):
        return {
            "event_type": self.event_type,
            "confidence": self.confidence,
            "arguments": [vars(argument) for argument in self.arguments]
        }
