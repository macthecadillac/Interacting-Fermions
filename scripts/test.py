# import numpy as np
from spinsys.hamiltonians.triangular_lattice_model import hamiltonian
from scipy import sparse
import spinsys


Nx = 4
Ny = 3
N = Nx * Ny
σz = spinsys.constructors.sigmaz()
z_mats = [spinsys.half.full_matrix(σz, k, N) for k in range(N)]

H = hamiltonian(Nx, Ny)
E, ground_state = sparse.linalg.eigsh(H, k=20, which='SA')
print(E)
