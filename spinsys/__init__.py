from spinsys import constructors
from spinsys import dmrg
from spinsys import exceptions
from spinsys import half
from spinsys import hamiltonians
from spinsys import quantities
from spinsys import state_generators
from spinsys import time_dependent
from spinsys import utils
import shutil
import numpy

__all__ = [
    "constructors",
    "dmrg",
    "exceptions",
    "half",
    "hamiltonians",
    "quantities",
    "state_generators",
    "time_dependent",
    "utils"
]

# set default print options for better display of data on screen
term_width = tuple(shutil.get_terminal_size())[0]
numpy.set_printoptions(precision=5, suppress=True, linewidth=term_width)
