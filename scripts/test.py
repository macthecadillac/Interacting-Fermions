# import numpy as np
from spinsys.hamiltonians.triangular_lattice_model import hamiltonian, SiteVector
from scipy import sparse
import spinsys


Nx = 4
Ny = 4
N = Nx * Ny
σz = spinsys.constructors.sigmaz()
z_mats = [spinsys.half.full_matrix(σz, k, N) for k in range(N)]
H = hamiltonian(Nx, Ny)
E, ground_state = sparse.linalg.eigsh(H, k=1, which='SA')

# # Test spin-spin correlation
state = sparse.csc_matrix(ground_state)
bonds = []
vec = SiteVector((0, 0), Nx, Ny)
for i in range(N):
    bonds.append((vec, vec.xhop(1)))
    bonds.append((vec, vec.yhop(1)))
    bonds.append((vec, vec.xhop(-1).yhop(1)))
    vec = vec.next_site()
for bond in bonds:
    site1, site2 = bond
    i, j = site1.lattice_index, site2.lattice_index
    corr = state.T.conjugate().dot(z_mats[i]).dot(z_mats[j]).dot(state)[0, 0]
    print("Bond = ({}, {}),   \tcorrelation = {}".format(i, j, round(corr.real, 14)))


# # Test entropy
print('\n')
state = state.toarray().flatten()
syss = [[0, 1, 4, 5], [1, 2, 5, 6], [5, 6, 9, 10], [6, 7, 10, 11], [8, 9, 12, 13], [10, 11, 14, 15], [13, 14, 1, 2], [14, 15, 2, 3]]
for sys in syss:
    reduced_ρ = spinsys.half.reduced_density_op(N, sys, state)
    entropy = spinsys.quantities.von_neumann_entropy(reduced_ρ)
    print(entropy)
