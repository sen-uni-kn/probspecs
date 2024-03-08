#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import typing
from abc import ABC
from copy import deepcopy

from frozendict import frozendict

from ruamel.yaml import yaml_object
from .yaml import yaml


@yaml_object(yaml)
class ConfigContainer(ABC):
    """
    Store configurations in an instance.

    Declare your configuration options in a class variable
    :code:`options`.
    :code:`ConfigContainer` handles setting these configurations and
    makes them accessible as instance attributes.
    For example, if you have an option :code:`opt`, you can access its value
    using `self.opt` or `getattr(self, "opt")` in any method of your subclass.
    """

    def __init__(self, **kwargs):
        """
        Creates a new :code:`ConfigContainer`.

        :param options: Some configurations.
        """
        config = {key: kwargs[key] for key in kwargs if key in self.config_keys}
        others = {key: kwargs[key] for key in kwargs if key not in self.config_keys}
        super().__init__(**others)
        self.__config = {}
        self.configure(**config)

    config_keys = frozenset({})

    @property
    def config(self) -> frozendict[str, typing.Any]:
        """The current configuration."""
        return frozendict(self.__config)

    def __getattr__(self, item):
        """Access a configuration option."""
        try:
            return self.__config[item]
        except KeyError:
            raise AttributeError()

    def configure(self, **kwargs):
        """
        Configures this instance.
        """
        for key in kwargs:
            if key not in self.config_keys:
                raise ValueError(f"Unknown configuration: {key}")
        self.__config.update(kwargs)

    def __copy__(self):
        return type(self)(**self.config)

    def __deepcopy__(self, memodict={}):
        config = deepcopy(self.config, memodict)
        return type(self)(**config)
