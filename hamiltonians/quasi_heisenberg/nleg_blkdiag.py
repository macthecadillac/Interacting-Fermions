"""Provides functions to generate the n-legged hamiltonian in the block
diagonal form.

3-18-2017
NOT YET USABLE
"""

import spinsys as s
from spinsys.utils.globalvar import Globals as G
import scipy.sparse as ss
import numpy as np
from itertools import zip_longest, chain


def diagonals(N, W1, c1, phi1, J1, W2, c2, phi2, J2, nleg, curr_j, mode):
    """Generates the diagonal elements of the hamiltonian.

    Args: "N" number of sites
          "W1" disorder strength along the "horizontal" direction
          "c1" trancendental number along the "horizontal" direction
          "phi1" phase along the "horizontal" direction
          "J1" coupling constant between sites along the "horizontal" direction
          "W2" disorder strength along the "vertical" direction
          "c2" trancendental number along the "vertical" direction
          "phi2" phase along the "verical" direction
          "J2" coupling constant between sites along the "verical" direction
          "nleg" total number of "legs" for the lattice
          "curr_j" total spin for the current block
          "mode" "open" or "periodic" boundary conditions
    Returns: csc_matrix
    """

    # cache to ram since the interaction contribution stays the same
    #  across configurations
    # @s.utils.io.cache_ram
    # this function doesn't use "curr_j", "J1", "J2" and "nleg" per se,
    # however, the inclusion of it in the arguments helps the caching function
    # identifying the data
    def interaction(N, J1, J2, nleg, curr_j, mode):
        diagonal = np.zeros(mat_dim)
        for i, st_config in enumerate(basis_set):
            # TODO PROBLEM IS HERE
            # convert state configuration from {0, 1} to {-1, 1}
            st_config = [conv[s] for s in st_config]
            # Interaction contributions. The conversion of 0's to -1's is
            # warranted by the use of multiplication here.  A basis state of
            # [1, 1, -1, -1] in <Sz_i Sz_(i+1)> would yield 1/4 - 1/4 + 1/4 -
            # 1/4 = 0. We can achieve the same effect here if we take 1/4 *
            # [1, 1, -1, -1] * [-1, 1, 1, -1], the last two terms being the
            # current state multiplied by its shifted self element wise. The
            # results are then summed.
            # --- along horizontal ---
            if mode == 'periodic' and not l1 == 2:
                x_interaction = sum(map(lambda x, y: x * y, st_config,
                                        st_config[-l2:] + st_config[:-l2]))
            else:
                x_interaction = sum(map(lambda x, y: x * y,
                                        st_config[l2:], st_config[:-l2]))
            # --- along vertical ---
            if mode == 'peiodic' and not l2 == 2:
                cols = map(list, zip_longest(*[iter(st_config)] * l2))
                pair = chain(*list(map(lambda x: [x[-1]] + x[:-1], cols)))
                y_interaction = sum(map(lambda x, y: x * y, st_config, pair))
            else:
                non_ledge_sites = [i for n, i in enumerate(st_config)
                                   if not n % l2 == 0]
                non_redge_sites = [i for n, i in enumerate(st_config)
                                   if not (n + 1) % l2 == 0]
                y_interaction = sum(map(lambda x, y: x * y, non_redge_sites,
                                        non_ledge_sites))
            diagonal[i] = 0.25 * (J1 * x_interaction + J2 * y_interaction)
        return diagonal

    l1 = N // nleg  # number of sites along the "horizontal"
    l2 = nleg       # number of sites along the "vertical" (number of legs)

    sites_1 = np.arange(1, l1 + 1)
    sites_2 = np.arange(1, l2 + 1)
    if l1 == 1:     # field contrib=0 if length of leg=1 aka 1D
        field1 = np.zeros([1])
    else:
        field1 = W1 * np.cos(2 * np.pi * c1 * sites_1 + phi1)
    if l2 == 1:      # field contrib=0 if length of leg=1 aka 1D
        field2 = np.zeros([1])
    else:
        field2 = W2 * np.cos(2 * np.pi * c2 * sites_2 + phi2)
    field = field1.repeat(l2) + field2.repeat(l1).reshape(l2, l1).T.flatten()

    basis_set = s.half.generate_complete_basis(N, curr_j)[0]
    conv = {1: 1, 0: -1}         # dict for conversion of 0's to -1's
    mat_dim = len(basis_set)     # size of the block for the given total <Sz>
    diagonal = np.zeros(mat_dim)
    for i, st_config in enumerate(basis_set):
        # convert 0's to -1's so the basis configuration would look like
        #  [-1, -1, 1, 1] instead of [0, 0, 1, 1]
        st_config = [conv[s] for s in st_config]
        # Disorder contributions.
        disord_contrib = sum(field * st_config)
        diagonal[i] = 0.5 * disord_contrib
    diagonal += interaction(N, J1, J2, nleg, curr_j, mode)
    ind = np.arange(0, mat_dim)
    return ss.csc_matrix((diagonal, (ind, ind)), shape=[mat_dim, mat_dim])


# @s.utils.io.cache_ram
# @s.utils.io.matcache
def off_diagonals(N, J1, J2, nleg, curr_j, mode):
    """Generate the off diagonal elements of the hamiltonian.

    Args: "N" total number of sites in the entire lattice
          "J1" coupling constant between adjacent sites along the "horizontal"
               direction
          "J2" coupling constant between adjacent sites along the "vertical"
               direction
          "nleg" number of "legs" in the lattice
          "curr_j" total spin of the system
          "mode" "periodic" or "open" boundary conditions
    Returns: csc_matrix
    """

    basis_set = s.half.generate_complete_basis(N, curr_j)[0]
    # The following section describes a hamiltonian of a 2-D latice that
    #  we label the following way
    #     0     l2      2*l2    ...  (l1-1)*l2
    #     1     l2+1    2*l2+1  ...  (l1-1)*l2+1
    #     2     l2+2    2*l2+2  ...  (l1-1)*l2+2
    #     ...
    #     l2-1  2*l2-1  3*l2-1  ...  l1*l2-1
    l1 = N // nleg  # number of sites along the "horizontal"
    l2 = nleg       # number of sites along the "vertical" (number of legs)
    lb1 = 0 if (mode == 'periodic' and not l1 == 2) else l2
    lb2 = 0 if (mode == 'periodic' and not l2 == 2) else 1

    # lattice indices labeled from 0 to N, filling the columns from left
    #  to right.
    lattice_indices = zip_longest(*[iter(range(N))] * l2)
    # pairs of adjacent elements to switch in a given basis for
    #  testing. This corresponds to nearest neighbor interaction
    adj_pairs_1 = [(j - l2, j) for j in range(lb1, N)]
    # This yields nonsence when l2=nleg=1 (self-interation). However,
    #  due to the nature of the raising and lowering operators, such
    #  hoppings automatically yield 0 and therefore does not change the
    #  results.
    adj_pairs_2 = [(col[i - 1], col[i]) for col in lattice_indices
                   for i in range(lb2, l2)]
    mat_dim = len(basis_set)    # size of the reduced Hilbert space
    hilbert_dim = 2 ** N        # size of the Hilbert space
    col_ind, row_ind, data = [], [], []
    # The "bra" in <phi|S_+S_- + S_-S_+|phi'>
    for J, adj_pairs in ((J1, adj_pairs_1), (J2, adj_pairs_2)):
        for i, bi in enumerate(basis_set):
            for pair in adj_pairs:
                bj = bi[:]          # the "ket" in <phi|S_+S_- + S_-S_+|phi'>
                bj[pair[0]], bj[pair[1]] = bj[pair[1]], bj[pair[0]]
                # Test to see if the pair of bases are indeed only different
                #  at two sites i.e. [1, 1, 0, 0] => [1, 0, 1, 0].
                #  If the premise is true, subtracting one basis from another
                #  element wise with each taken the absolute value would yield
                #  2.
                if sum(map(lambda x, y: abs(x - y), bi, bj)) == 2:
                    j_loc = hilbert_dim - s.utils.misc.bin_to_dec(bj) - 1
                    j = G['complete_basis'][N][curr_j][1][j_loc]
                    col_ind.append(j)
                    row_ind.append(i)
                    data.append(0.5 * J)
    return ss.csc_matrix((data, (row_ind, col_ind)), shape=[mat_dim, mat_dim])


def H(N, W1, c1, phi1, J1=1, W2=0, c2=0, phi2=0, J2=0, nleg=1,
      curr_j=0, mode='open'):
    """Generates the full hamiltonian for a block corresponding to a
    specific total spin.

    Args: "N" total number of sites
          "W1" disorder strength along the horizontal
          "c1" trancendental number in the cosine term in the horizontal
               field term
          "phi1" phase factor for the horizontal field term
          "J1" nearest neighbor coupling constant along the horizontal
          "W2" disorder strength along the vertical
          "c2" trancendental number in the cosine term in the vertical
               field term
          "phi2" phase factor for the vertical field term
          "J2" nearest neighbor coupling constant along the vertical
          "nleg" number of legs i.e. number of horizontal rungs on the
                 lattice
          "mode" "peiodic" or "open" boundary conditions
    Returns: csc_matrix
    """
    return diagonals(N, W1, c1, phi1, J1, W2, c2, phi2, J2, nleg, curr_j,
                     mode) + \
           off_diagonals(N, J1, J2, nleg, curr_j, mode)
