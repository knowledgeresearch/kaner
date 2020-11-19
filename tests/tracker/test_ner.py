# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""NERTracker tests"""

import tempfile
import os
from kaner.tracker import NERTrackerRow, NERTracker


def test_nertracker():
    """Test the class `NERTracker`."""
    tmp_folder = tempfile.TemporaryDirectory()
    folder_name = tmp_folder.name
    tracker = NERTracker(NERTrackerRow)
    ground_truth = 'date,model,dataset,tokenizer_model,gazetteer_model,log_dir,f1_score,precision_score,'\
                   + 'recall_score,time_consu,total_epoch,test_loss,time_per_epoch,tag'\
                   + '\n2020-08-22,model_test,dataset_test,t1,g1,l1,0.1,0.2,0.3,1000.0,10,0.1,99.999990000001,nil\n'
    row = NERTrackerRow(
        "2020-08-22", "model_test", "dataset_test", "t1", "g1", "l1", 1000.0, 0.1, 0.2, 0.3, 10, 0.1, "nil"
    )
    labpath = os.path.join(folder_name, "nertracker_test")
    tracker.insert(row)
    tracker.save(labpath)
    with open(labpath, "r") as f_in:
        content = f_in.read()
    assert content == ground_truth
    tracker.insert(row)
    results = [
        {
            'dataset': 'dataset_test',
            'f1_score': 0.1,
            'gazetteer_model': 'g1',
            'model': 'model_test',
            'n_experiments': 2,
            "tag": "nil",
            'precision_score': 0.2,
            'recall_score': 0.3,
            'test_loss': 0.1,
            'time_consu': 1000.0,
            'time_per_epoch': 99.999990000001,
            'tokenizer_model': 't1',
            'total_epoch': 10.0
        }
    ]
    assert tracker.summay() == results
    tmp_folder.cleanup()
