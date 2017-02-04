"""This module provides custom exception classes for more convenient
exception handling

1-13-2017
"""


class NoConvergence(Exception):
    pass


class SizeMismatchError(Exception):
    pass


class StateNotFoundError(Exception):
    pass
