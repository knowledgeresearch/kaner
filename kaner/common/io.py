# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""IO utils"""

import os
import json
from typing import Tuple, List
from collections import OrderedDict
import yaml

import xmltodict


def load_xml_as_json(encoding: str, *file_path: Tuple[str]) -> dict:
    """
    Convert XML file to a dictionary.

    References:
        [1] https://www.geeksforgeeks.org/python-xml-to-json/

    Args:
        file_path (tuple): The file path.
        encoding (str): The encoding of the given file.
    """
    def convert(data: dict):
        """
        Convert OrderedDict to dict.
        """
        for key, obj in data.items():
            if isinstance(obj, OrderedDict):
                data[key] = dict(obj)
                convert(data[key])
    file_path = os.path.join(*file_path)
    with open(file_path, "r", encoding=encoding) as f_in:
        content = f_in.read()
    data = dict(xmltodict.parse(content))
    convert(data)

    return data


def load_yaml_as_json(encoding: str, *file_path: Tuple[str]) -> dict:
    """
    Convert YAML file to a dictionary.

    References:
        [1] https://www.geeksforgeeks.org/python-xml-to-json/

    Args:
        file_path (tuple): The file path.
        encoding (str): The encoding of the given file.
    """
    file_path = os.path.join(*file_path)
    with open(file_path, "r", encoding=encoding) as f_in:
        data = yaml.load(f_in, Loader=yaml.FullLoader)

    return data


def load_json(encoding: str, *file_path: Tuple[str]) -> dict:
    """
    Convert json file to a dictionary.

    Args:
        file_path (tuple): The file path.
        encoding (str): The encoding of the given file.
    """
    file_path = os.path.join(*file_path)
    with open(file_path, "r", encoding=encoding) as f_in:
        content = f_in.read()
    data = json.loads(content)

    return data


def save_json(data: dict, *file_path: Tuple[str]):
    """
    Save a dictionary to a given path.

    Args:
        data (dict): The dictionary to be saved.
        file_path (tuple): The file path.
    """
    file_path = os.path.join(*file_path)
    with open(file_path, "w", encoding="utf-8") as f_out:
        f_out.write(json.dumps(data, ensure_ascii=False))


def load_jsonl(encoding: str, *file_path: Tuple[str]) -> list:
    """
    Convert jsonl file to a list.

    Args:
        file_path (tuple): The file path.
        encoding (str): The encoding of the given file.
    """
    data = []
    file_path = os.path.join(*file_path)
    with open(file_path, "r", encoding=encoding) as f_in:
        for line in f_in.readlines():
            data.append(json.loads(line))

    return data


def save_jsonl(data: List[dict], *file_path: Tuple[str]):
    """
    Save a list of dictionary to a given path.

    Args:
        data (List[dict]): A list of dictionary to be saved.
        file_path (tuple): The file path.
    """
    lines = [json.dumps(ins, ensure_ascii=False) for ins in data]
    file_path = os.path.join(*file_path)
    with open(file_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))


def save_text(text: str, *file_path: Tuple[str]):
    """
    Save text to a given path.

    Args:
        text (str): Text to be saved.
        file_path (tuple): The file path.
    """
    file_path = os.path.join(*file_path)
    with open(file_path, "w", encoding="utf-8") as f_out:
        f_out.write(text)
