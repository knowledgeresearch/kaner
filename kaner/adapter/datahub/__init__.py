# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Dataset hubs for preprocessing datasets to a uniform format"""

__all__ = [
    "MSRANER",
    "WeiboNER",
    "OntoNotes",
    "ResumeNER",
    "ECommerce",
    "CCKSEE",
    "CHIP"
]

from .msraner import MSRANER
from .weiboner import WeiboNER
from .ontonotes import OntoNotes
from .resumener import ResumeNER
from .ecommerce import ECommerce
from .ccksee import CCKSEE
from .chip import CHIP
