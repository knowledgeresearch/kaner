# coding=utf-8
# Copyright (c) Knowledge Research and its affiliates. All Rights Reserved
"""Context Management"""

from typing import Callable, Any, List
import inspect


class GlobalContext:
    """
    GlobalContext provides a global context management for conveniently accessing various datasets, adapters, and models.
    """

    __datahub = {}
    __tokenizers = {}
    __gazetteers = {}
    __in_adapters = {}
    __batcher = {}
    __train_callback = {}
    __test_callback = {}
    __models = {}
    __out_adapters = {}

    @staticmethod
    def init():
        """
        Initialize all classes and functions.
        """
        from kaner import model as _MODEL
        from kaner.trainer import callback as _CALLBACK
        from kaner.adapter import (
            in_adapter as _IN_ADAPTER,
            out_adapter as _OUT_ADAPTER,
            batcher as _BATCHER,
            datahub as _DATAHUB,
            modelhub as _MODELHUB
        )

    @classmethod
    def get_dataset_names(cls) -> List[str]:
        """
        Return all registered dataset's name in datahub.
        """
        return list(cls.__datahub.keys())

    @classmethod
    def get_tokenizer_names(cls) -> List[str]:
        """
        Return all registered tokenizer model's name in modelhub.
        """
        return list(cls.__tokenizers.keys())

    @classmethod
    def get_gazetteer_names(cls) -> List[str]:
        """
        Return all registered gazetteer model's name in modelhub.
        """
        return list(cls.__gazetteers.keys())

    @classmethod
    def get_model_names(cls) -> List[str]:
        """
        Return all registered model's name in models.
        """
        return list(cls.__models.keys())

    @classmethod
    def register_datahub(cls, uid: str) -> Callable:
        """
        Register a datahub using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_class: Any) -> Callable:
            if uid not in cls.__datahub.keys():
                cls.__datahub[uid] = wrapper_class
            return wrapper_class
        return wrapper

    @classmethod
    def create_datahub(cls, uid: str, **class_args) -> Any:
        """
        Create a datahub instance using its unique identifier.

        Args:
            uid (str): Unique identifier.
            class_args (dict): Class arguments.
        """
        if uid not in cls.__datahub.keys():
            return None
        datahub_cls = cls.__datahub[uid]
        args = inspect.getfullargspec(datahub_cls.__init__).args
        kwargs = {}
        for arg in args:
            if arg in class_args.keys():
                kwargs[arg] = class_args[arg]
        return datahub_cls(**kwargs)

    @classmethod
    def register_tokenizer(cls, uid: str) -> Callable:
        """
        Register a tokenizer model using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_class: Any) -> Callable:
            if uid not in cls.__tokenizers.keys():
                cls.__tokenizers[uid] = wrapper_class
            return wrapper_class
        return wrapper

    @classmethod
    def create_tokenizer(cls, uid: str, **class_args) -> Any:
        """
        Create a tokenizer model instance using its unique identifier.

        Args:
            uid (str): Unique identifier.
            class_args (dict): Class arguments.
        """
        if uid not in cls.__tokenizers.keys():
            return None
        tokenizer_cls = cls.__tokenizers[uid]
        args = inspect.getfullargspec(tokenizer_cls.__init__).args
        kwargs = {}
        for arg in args:
            if arg in class_args.keys():
                kwargs[arg] = class_args[arg]
        return tokenizer_cls(**kwargs)

    @classmethod
    def register_gazetteer(cls, uid: str) -> Callable:
        """
        Register a gazetteer model using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_class: Any) -> Callable:
            if uid not in cls.__gazetteers.keys():
                cls.__gazetteers[uid] = wrapper_class
            return wrapper_class
        return wrapper

    @classmethod
    def create_gazetteer(cls, uid: str, **class_args) -> Any:
        """
        Create a gazetteer model instance using its unique identifier.

        Args:
            uid (str): Unique identifier.
            class_args (dict): Class arguments.
        """
        if uid not in cls.__gazetteers.keys():
            return None
        gazetteer_cls = cls.__gazetteers[uid]
        args = inspect.getfullargspec(gazetteer_cls.__init__).args
        kwargs = {}
        for arg in args:
            if arg in class_args.keys():
                kwargs[arg] = class_args[arg]
        return gazetteer_cls(**kwargs)

    @classmethod
    def register_inadapter(cls, uid: str) -> Callable:
        """
        Register an input adapter using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_class: Any) -> Callable:
            if uid not in cls.__in_adapters.keys():
                cls.__in_adapters[uid] = wrapper_class
            return wrapper_class
        return wrapper

    @classmethod
    def create_inadapter(cls, uid: str, **class_args) -> Any:
        """
        Create an input adapter instance using its unique identifier.

        Args:
            uid (str): Unique identifier.
            class_args (dict): Class arguments.
        """
        if uid not in cls.__in_adapters.keys():
            return None
        inadapter_cls = cls.__in_adapters[uid]
        args = inspect.getfullargspec(inadapter_cls.__init__).args
        kwargs = {}
        for arg in args:
            if arg in class_args.keys():
                kwargs[arg] = class_args[arg]
        return inadapter_cls(**kwargs)

    @classmethod
    def register_outadapter(cls, uid: str) -> Callable:
        """
        Register an output adapter using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_class: Any) -> Callable:
            if uid not in cls.__out_adapters.keys():
                cls.__out_adapters[uid] = wrapper_class
            return wrapper_class
        return wrapper

    @classmethod
    def create_outadapter(cls, uid: str, **class_args) -> Any:
        """
        Create an output adapter instance using its unique identifier.

        Args:
            uid (str): Unique identifier.
            class_args (dict): Class arguments.
        """
        if uid not in cls.__out_adapters.keys():
            return None
        outadapter_cls = cls.__out_adapters[uid]
        args = inspect.getfullargspec(outadapter_cls.__init__).args
        kwargs = {}
        for arg in args:
            if arg in class_args.keys():
                kwargs[arg] = class_args[arg]
        return outadapter_cls(**kwargs)

    @classmethod
    def register_model(cls, uid: str) -> Callable:
        """
        Register a model using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_class: Any) -> Callable:
            if uid not in cls.__models.keys():
                cls.__models[uid] = wrapper_class
            return wrapper_class
        return wrapper

    @classmethod
    def create_model(cls, uid: str, **class_args) -> Any:
        """
        Create a model instance using its unique identifier.

        Args:
            uid (str): Unique identifier.
            class_args (dict): Class arguments.
        """
        if uid not in cls.__models.keys():
            return None
        model_cls = cls.__models[uid]
        args = inspect.getfullargspec(model_cls.__init__).args
        kwargs = {}
        for arg in args:
            if arg in class_args.keys():
                kwargs[arg] = class_args[arg]
        return model_cls(**kwargs)

    @classmethod
    def register_batcher(cls, uid: str) -> Callable:
        """
        Register a batch collate function using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_func: Any) -> Callable:
            if uid not in cls.__batcher.keys():
                cls.__batcher[uid] = wrapper_func
            return wrapper_func
        return wrapper

    @classmethod
    def create_batcher(cls, uid: str, **func_args) -> Any:
        """
        Create a batch collate function using its unique identifier.

        Args:
            uid (str): Unique identifier.
            func_args (dict): Function arguments.
        """
        if uid not in cls.__batcher.keys():
            return None
        wrapper_func = cls.__batcher[uid]
        args = inspect.getfullargspec(wrapper_func).args
        kwargs = {}
        for arg in args:
            if arg in func_args.keys():
                kwargs[arg] = func_args[arg]
        return wrapper_func(**kwargs)

    @classmethod
    def register_traincallback(cls, uid: str) -> Callable:
        """
        Register a train callback function using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_func: Any) -> Callable:
            if uid not in cls.__train_callback.keys():
                cls.__train_callback[uid] = wrapper_func
            return wrapper_func
        return wrapper

    @classmethod
    def create_traincallback(cls, uid: str) -> Any:
        """
        Create a train callback function using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        return cls.__train_callback.get(uid, None)

    @classmethod
    def register_testcallback(cls, uid: str) -> Callable:
        """
        Register a test callback function using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        def wrapper(wrapper_func: Any) -> Callable:
            if uid not in cls.__test_callback.keys():
                cls.__test_callback[uid] = wrapper_func
            return wrapper_func
        return wrapper

    @classmethod
    def create_testcallback(cls, uid: str) -> Any:
        """
        Create a test callback function using its unique identifier.

        Args:
            uid (str): Unique identifier.
        """
        return cls.__test_callback.get(uid, None)
