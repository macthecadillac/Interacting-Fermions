from scipy import sparse
import numpy as np
import spinsys
from hamiltonians.triangular_lattice_model import hamiltonian


def translational_operator(Nx, Ny, direction='x'):
    N = Nx * Ny
    format_options = '0{}b'.format(N)
    old_ind = list(range(2 ** N))
    original_basis = list(map(lambda x: format(x, format_options), old_ind))
    partitioned = [[i[y * Nx:(y + 1) * Nx] for y in range(Ny)] for i in original_basis]
    print(partitioned)
    if direction == 'x':
        new_basis = []
        for basis_state in partitioned:
            new_basis.append([i[-1] + i[:-1] for i in basis_state])
        # translated_basis = [i[-1] + i[:-1] for i in original_basis]
        # new_ind = list(map(lambda x: int(x, 2), translated_basis))
        # A = sparse.csc_matrix((np.ones(2 ** N), (old_ind, new_ind)))
        # print(new_ind)
        # print(A.toarray())
    print(new_basis)


Nx = 2
Ny = 3
N = Nx * Ny
H = hamiltonian(Nx, Ny)
E, ψ = sparse.linalg.eigsh(H, k=1, which='SA')
translational_operator(Nx, Ny)
