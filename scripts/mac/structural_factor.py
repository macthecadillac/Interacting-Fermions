import numpy as np
from scipy import sparse
from spinsys import utils, quantities
from hamiltonians.triangular_lattice_model import hamiltonian


Nx = 4
Ny = 4
N = Nx * Ny
H = hamiltonian(Nx, Ny)
E, ψ0 = sparse.linalg.eigsh(H, k=1, which='SA')
ψ0 = sparse.csc_matrix(ψ0)
nkx = nky = 500
data = np.empty((nkx, nky), dtype=complex)
t = utils.timer.Timer(nkx * nky)
for i, kx in enumerate(np.linspace(-np.pi, np.pi, nkx, endpoint=False)):
    for j, ky in enumerate(np.linspace(-np.pi, np.pi, nky, endpoint=False)):
        s = quantities.structural_factor(Nx, Ny, kx, ky, ψ0)
        data[i, j] = s
        t.progress()
np.save('/home/mac/Sync/Research Data/Triangular lattice result/structural_factor_{}x{}.npy'
        .format(Nx, Ny), data, allow_pickle=False)
