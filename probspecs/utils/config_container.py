#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import typing
from abc import ABC

from frozendict import frozendict


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

    def __init__(self, **options):
        """
        Creates a new :code:`ConfigContainer`.

        :param options: Some configurations.
        """
        self.__config = {}
        self.configure(**options)

    options = frozenset({})

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
            if key not in self.options:
                raise ValueError(f"Unknown configuration: {key}")
        self.__config.update(kwargs)
