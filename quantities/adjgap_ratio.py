"""This module provides functions related to the calculation of the
adjacent gap ratio

1-13-2017
"""

import numpy as np


def adj_gap_ratio(sorted_eigvals):
    """
    Takes a list of eigenvalues that have been sorted low to high, finds the
    adjacent gap ratio for each set of 3 adjacent eigenvalues
    :param sorted_eigvals:
    :return adj_gap_ratio:
    """
    deltas = np.diff(sorted_eigvals)
    agrs = [min(deltas[i], deltas[i + 1]) / max(deltas[i], deltas[i + 1])
            for i in range(len(deltas) - 1)]
    return np.array(agrs)
