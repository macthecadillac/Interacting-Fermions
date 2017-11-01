import numpy as np
from scipy import sparse
from spinsys.utils import timer
from hamiltonians.triangular_lattice_model import hamiltonian


Nx = 4
Ny = 4
N = Nx * Ny
spectra = np.empty((30, 20))
t = timer.Timer(20)
for col, J_ppmm in enumerate(np.linspace(0, 10, 20)):
    H = hamiltonian(Nx, Ny, J_ppmm=J_ppmm)
    Es = sparse.linalg.eigsh(H, k=30, which='SA', return_eigenvectors=False)
    spectra[:, col] = Es
    t.progress()
np.savetxt('spectra_J++--.txt', spectra)
