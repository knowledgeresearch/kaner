# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Package"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="kaner",
    version="0.0.1",
    author="Knowledge Research",
    description="A toolkit for Knowledge-Aware Named Entity Recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knowledgeresearch/kaner",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
