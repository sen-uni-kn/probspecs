# Copyright (c) 2023 David Boetius
# Licensed under the MIT license


def contains_unbracketed(string: str, symbols: tuple[str, ...]):
    """
    Returns whether a string contains certain symbols on the top-most
    level, that is, without being surrounded by brackets.

    For example, :code:`a + b * (c - d)` contains :code:`+`
    and :code:`*` on the top-most level, but not :code:`-`
    """
    bracket_level = 0
    for char in string:
        if char in ("(", "[", "{"):
            bracket_level += 1
        elif char in (")", "]", "}"):
            bracket_level -= 1
        if bracket_level == 0 and char in symbols:
            return True
    return False
