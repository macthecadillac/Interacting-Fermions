import numpy as np
import pathlib
from scipy import sparse
from spinsys import utils, quantities
from hamiltonians.triangular_lattice_model import hamiltonian_dp


home = str(pathlib.Path.home())
Nx = 4
Ny = 4
N = Nx * Ny
nkx = nky = 500
J_z = 1
J_pm = 0.5
t = utils.timer.Timer(nkx * nky * 21)
for J_ppmm in np.linspace(-1, 1, 21):
    H = hamiltonian_dp(Nx, Ny, J_pm=J_pm, J_z=J_z, J_ppmm=J_ppmm)
    E, ψ0 = sparse.linalg.eigsh(H, k=1, which='SA')
    ψ0 = sparse.csc_matrix(ψ0)
    data = np.empty((nkx, nky), dtype=complex)
    for i, kx in enumerate(np.linspace(-np.pi, np.pi, nkx, endpoint=False)):
        for j, ky in enumerate(np.linspace(-np.pi, np.pi, nky, endpoint=False)):
            s = quantities.structural_factor(Nx, Ny, kx, ky, ψ0)
            data[i, j] = s
            t.progress()
    np.save('{}/structural_factor_{}x{}_Jz{}_J+-{}_J++--{}.npy'
            .format(home, Nx, Ny, J_z, J_pm, J_ppmm), data, allow_pickle=False)
