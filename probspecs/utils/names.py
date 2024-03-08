#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
from abc import ABC
from random import randint

from randomname import get_name


class Named(ABC):
    """
    Provides non-necessarily unique names to objects.
    The names are chosen randomly from a large pool of combinations,
    so collisions are unlikely.
    """

    def __init__(self, **kwargs):
        self.__name = get_name(
            adj=(
                "character",
                "corporate_prefixes",
                "emotions",
                "shape",
                "size",
                "speed",
                "temperature",
                "weather",
            ),
            noun=(
                "birds",
                "cats",
                "dogs",
                "wood",
                "minerals",
                "fruit",
                "ghosts",
                "music_instruments",
                "fish",
                "plants",
            ),
            seed=randint(0, 1000) + id(self),
        )
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return self.__name
