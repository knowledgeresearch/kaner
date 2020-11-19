# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Service"""

import json
import os
from datetime import datetime

from flask import Flask
from flask import request
from flask import Response
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from kaner.context import GlobalContext as gctx
from kaner.adapter.tokenizer import CharTokenizer
from kaner.adapter.knowledge import Gazetteer
from kaner.trainer import NERTrainer, TrainerConfig
from kaner.common import load_json


gctx.init()
__APP__ = Flask(__name__)
__TRAINER__ = None


@__APP__.route("/", methods=["GET"])
def _hello():
    """Return welcome!"""
    return "<h1>Welcome to KANER!</h1>"


@__APP__.route("/kaner/predict", methods=["POST", "OPTIONS"])
def _predict_span():
    """
    Given a list of text, predicting their spans. The format of the requested data should be as the same as
    the following example:

        data = {"texts": ["document1", "document2", ...]}
    """
    start_time = datetime.now()
    rsp = {
        "success": True
    }
    try:
        data = json.loads(request.data)
        if "texts" not in data.keys() or not isinstance(data["texts"], list):
            rsp["success"] = False
            rsp["info"] = "The requested data should contain a list of document named `texts`."
            return json.dumps(rsp, ensure_ascii=False, indent=2)
        if len(data["texts"]) <= 0:
            rsp["success"] = False
            rsp["info"] = "The length of the list `texts` should be greater than 0"
            return json.dumps(rsp, ensure_ascii=False, indent=2)
        for i in range(len(data["texts"])):
            text = data["texts"][i]
            if not isinstance(text, str) or len(text) == 0:
                rsp["success"] = False
                rsp["info"] = "The {0}-th element in the list is not a string or is empty. {1}。".format(i + 1, text)
                return json.dumps(rsp, ensure_ascii=False, indent=2)
        if "debug" not in data.keys():
            data["debug"] = False
        global __TRAINER__
        if __TRAINER__ is None:
            rsp["success"] = False
            rsp["info"] = "The server is not prepared. Please wait a while..."
        else:
            results = []
            for _, text in enumerate(data["texts"]):
                fragments = []
                for i in range(0, len(text), __TRAINER__._config.max_seq_len):
                    fragments.append(text[i: i + __TRAINER__._config.max_seq_len])
                raw_result = __TRAINER__.predict(fragments)
                result = {
                    "text": text,
                    "spans": []
                }
                for i in range(0, len(text), __TRAINER__._config.max_seq_len):
                    idx = i//__TRAINER__._config.max_seq_len
                    for j in range(len(raw_result[idx]["spans"])):
                        raw_result[idx]["spans"][j]["start"] += i
                        raw_result[idx]["spans"][j]["end"] += i
                    result["spans"].extend(raw_result[idx]["spans"])
                results.append(result)
            rsp["data"] = results
            rsp["model"] = __TRAINER__._config.model
            rsp["dataset"] = __TRAINER__._config.dataset
    except RuntimeError as runtime_error:
        rsp["success"] = False
        rsp["info"] = "Runtime Error {}.".format(str(runtime_error))
    except json.decoder.JSONDecodeError as json_decode_error:
        rsp["success"] = False
        rsp["info"] = "Parsing JSON String Error {}.".format(str(json_decode_error))
    except Exception as exception:
        rsp["success"] = False
        rsp["info"] = str(exception)
    rsp["time"] = (datetime.now() - start_time).seconds
    rsp = Response(json.dumps(rsp, ensure_ascii=False, indent=2))
    rsp.headers["content-type"] = "application/json"
    # A CORS preflight request is a CORS request that checks to see if the CORS protocol is
    # understood and a server is aware using specific methods and headers.
    # https://developer.mozilla.org/en-US/docs/Glossary/Preflight_request
    rsp.headers["Access-Control-Allow-Origin"] = "*"
    rsp.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept"

    return rsp


def serve(model_folder: str, host: str, port: int) -> None:
    """
    Start a service for predicting texts.

    Args:
        model_folder (str): The folder where the trained model exists.
        host (str): Address.
        port (str): Listen port.
    """
    options = load_json("utf-8", model_folder, "config.json")
    options["output_folder"] = model_folder
    options["identity"] = os.path.basename(os.path.normpath(model_folder))
    config = TrainerConfig(options)
    tokenizer = CharTokenizer(model_folder)
    gazetteer = Gazetteer(model_folder)
    out_adapter = gctx.create_outadapter(config.out_adapter, dataset_folder=model_folder, file_name="labels")
    in_adapters = (
        gctx.create_inadapter(
            config.in_adapter, dataset=[], tokenizer=tokenizer, out_adapter=out_adapter, gazetteer=gazetteer,
            **config.hyperparameters
        )
        for _ in range(3)
    )
    collate_fn = gctx.create_batcher(
        config.model, input_pad=tokenizer.pad_id, output_pad=out_adapter.unk_id, lexicon_pad=gazetteer.pad_id, device=config.device
    )
    model = gctx.create_model(config.model, **config.hyperparameters)
    trainer = NERTrainer(
        config, tokenizer, in_adapters, out_adapter, collate_fn, model, None,
        gctx.create_traincallback(config.model), gctx.create_testcallback(config.model)
    )
    trainer.startp()
    global __TRAINER__, __APP__
    __TRAINER__ = trainer
    server = HTTPServer(WSGIContainer(__APP__))
    print("->    :: Web Service ::    ")
    print("service starts at \x1b[1;32;40mhttp://{0}:{1}\x1b[0m".format(host, port))
    server.listen(port, host)
    IOLoop.instance().start()
