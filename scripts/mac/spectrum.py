import numpy as np
from scipy import sparse
from spinsys.utils import timer
from hamiltonians.triangular_lattice_model import hamiltonian


Nx = 4
Ny = 4
N = Nx * Ny
nstates = 1000
neigs = 15
spectra = np.empty((neigs, nstates))
t = timer.Timer(20)
for col, J_ppmm in enumerate(np.linspace(0, 10, nstates)):
    H = hamiltonian(Nx, Ny, J_ppmm=J_ppmm)
    Es = sparse.linalg.eigsh(H, k=neigs, which='SA', return_eigenvectors=False)
    spectra[:, col] = Es
    t.progress()
np.save('spectra_J++--.npy', spectra, allow_pickle=False)
