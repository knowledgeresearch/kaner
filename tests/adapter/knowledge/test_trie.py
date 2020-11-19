# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Trie tests"""

from kaner.adapter.knowledge.trie import TrieTree


def test_trie():
    """Test the module `trie`."""
    sentence = "南京市长江大桥"
    trie = TrieTree()
    assert trie.search(["南", "京"]) is False
    trie.insert(["南", "京"])
    assert trie.search(["南", "京"]) is True
    assert trie.search(["南"]) is False
    trie.insert(["南", "京", "市"])
    assert trie.search(["南", "京"]) is True
    assert trie.search(["南"]) is False
    assert trie.search(["南", "京", "市"]) is True
    assert trie.search(["南", "京", "市", "长", "江", "大", "桥"]) is False
    trie.insert(["南", "京", "大", "学"])
    assert trie.enumerate_match(list(sentence)) == ["南京", "南京市"]
    # test mask
    sentence = "南京市长江大桥"
    trie = TrieTree()
    trie.insert(["南", "京"])
    trie.insert(["南", "京", "市"])
    trie.insert(["长", "江"])
    trie.insert(["长", "江", "大", "桥"])
    trie.mask(["南京市", "长江大桥"])
    assert trie.enumerate_match(list(sentence)) == ["南京"]
    assert trie.enumerate_match(list(sentence[3:])) == ["长江"]
    trie.clear_mask(["南京市", "长江大桥"])
    assert trie.enumerate_match(list(sentence)) == ["南京", "南京市"]
    assert trie.enumerate_match(list(sentence[3:])) == ["长江", "长江大桥"]
