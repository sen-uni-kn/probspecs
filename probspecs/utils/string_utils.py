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


def item_to_str(item: int | slice | tuple[int | slice, ...]):
    """
    Formats the argument of :code:`__getitem__` in a nice way, primarily
    formatting slices the way they are written in code.
    For example, :code:`1:` becomes `"1:"` instead of `"slice(1, None, None)"`.
    """
    if isinstance(item, tuple):
        return ", ".join(item_to_str(elem) for elem in item)
    elif isinstance(item, slice):
        item_str = ""
        if item.start is not None:
            item_str += str(item.start)
        item_str += ":"
        if item.stop is not None:
            item_str += str(item.stop)
        if item.step is not None:
            item_str += ":"
            item_str += str(item.step)
        return item_str
    else:
        return str(item)
