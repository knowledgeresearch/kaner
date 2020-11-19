# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""IO utils tests"""

import tempfile
import os

from kaner.common.io import (
    load_xml_as_json,
    load_json,
    load_jsonl,
    load_yaml_as_json,
    save_json,
    save_jsonl
)


def test_func():
    """Test the module `io`."""
    content_json = {
        "id": "1",
        "data": {
            "test": ["1", "2"]
        }
    }
    content_yml = "id: '1'\ndata:\n   test: ['1', '2']"
    content_xml = '<?xml version="1.0" encoding="UTF-8"?>\
        <root><data><test>1</test><test>2</test></data><id>1</id></root>'
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    # test save_load_json
    save_json(content_json, os.path.join(folder_name, "save_load_json"))
    assert load_json("utf-8", folder_name, "save_load_json") == content_json
    # test load_xml_as_json
    with open(os.path.join(folder_name, "load_xml_as_json"), "w") as f_out:
        f_out.write(content_xml)
    assert load_xml_as_json("utf-8", f_out.name)["root"] == content_json
    # test load_yaml_as_json
    with open(os.path.join(folder_name, "load_yaml_as_json"), "w") as f_out:
        f_out.write(content_yml)
    assert load_yaml_as_json("utf-8", f_out.name) == content_json
    # save_load_jsonl
    save_jsonl([content_json, content_json], os.path.join(folder_name, "save_load_jsonl"))
    assert all([ins == content_json for ins in load_jsonl("utf-8", folder_name, "save_load_jsonl")])
    tmp_folder.cleanup()
