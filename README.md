# Dependencies
The following packages will need to be installed: numpy, scipy, msgpack-python 

# Installation
Copy and paste the following in the terminal
```shell
git clone https://github.com/1119group/spinsys.git
```
You will now find the folder in your home directory. To make the library available 
in the prompt, link the folder to the local Python path. On Linux, the folder
should be linked to 
~/.local/lib/python3.x/site-packages/ where x is the current default Python
version on your system. Once that is done, the library is available system-wide
(for your user) and you can import it the usual way.

# Usage
The library can be imported with
```python
import spinsys
```
It is preferable that modules within the spinsys folder be left unmodified unless the
function written is general enough to be sorted into one of the
sub-modules/sub-folders. Temporary scripts for plotting and simulation should
be put somewhere else to keep the library/repository clean and well-organized.

# Included in the package:
## Hamiltonians
    spinsys.hamiltonians.quasi_heisenberg.oneleg.H
    spinsys.hamiltonians.quasi_heisenberg.oneleg_blkdiag.H
    spinsys.hamiltonians.quasi_heisenberg.nleg.H
    spinsys.hamiltonians.quasi_heisenberg.nleg_blkdiag.H
    spinsys.hamiltonians.heisenberg.oneleg.H
    spinsys.hamiltonians.heisenberg.oneleg_blkdiag.H

## Functions
    spinsys.constructors.raising
    spinsys.constructors.lowering
    spinsys.constructors.sigmax
    spinsys.constructors.sigmay
    spinsys.constructors.sigmaz
    spinsys.half.generate_complete_basis
    spinsys.half.full_matrix
    spinsys.half.expand_and_reorder
    spinsys.half.similarity_trans_matrix
    spinsys.quantities.adj_gap_ratio
    spinsys.quantities.bipartite_reduced_density_op
    spinsys.quantities.spin_glass_order
    spinsys.quantities.von_neumann_entropy
    spinsys.state_generators.generate_eigenpairs
    spinsys.state_generators.generate_product_state
    spinsys.utils.io.cache
    spinsys.utils.io.matcache
    spinsys.utils.io.cache_ram
    spinsys.utils.log.logged
    spinsys.utils.misc.bin_to_dec
    spinsys.utils.misc.permutation
    spinsys.utils.misc.binary_permutation
    spinsys.utils.misc.permutations_any_order

## Classes
    spinsys.exceptions.NoConvergence
    spinsys.exceptions.SizeMismatchError
    spinsys.exceptions.StateNotFoundError
    spinsys.time_dependent.TimeMachine
    spinsys.utils.timer.Timer
