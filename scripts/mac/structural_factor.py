import numpy as np
# import pathlib
from scipy import sparse
from spinsys import quantities
from hamiltonians.triangular_lattice_model import hamiltonian_dp


def structural_factor(Nx, Ny, kx, ky, ψ):
    pass


# # home = str(pathlib.Path.home())
Nx = 4
Ny = 4
N = Nx * Ny
J_z = 1
J_pm = 0.5
J_ppmm = 0.1
H = hamiltonian_dp(Nx, Ny, J_pm=J_pm, J_z=J_z, J_ppmm=J_ppmm)
E, ψ0 = sparse.linalg.eigsh(H, k=1, which='SA')
ψ0 = sparse.csc_matrix(ψ0)
data = np.empty((N, 3), dtype=complex)
for i in range(0, Nx):
    for j in range(0, Ny):
        kx = 2 * np.pi * i / Nx
        ky = 2 * np.pi * j / Ny
        s = quantities.structural_factor(Nx, Ny, kx, ky, ψ0)
        col = i * Nx + j
        data[col, 0] = i
        data[col, 1] = j
        data[col, 2] = s
# np.save('{}/structural_factor_{}x{}_Jz{}_J+-{}_J++--{}.npy'
#         .format(home, Nx, Ny, J_z, J_pm, J_ppmm), data, allow_pickle=False)
print(data)

# sz_corr = np.array([
#     [0.0000000, 0.0000000, 0.2500000],
#     [1.0000000, 0.0000000, -0.0552257],
#     [2.0000000, 0.0000000, 0.0227398],
#     [3.0000000, 0.0000000, -0.0552257],
#     [0.0000000, 1.0000000, -0.0552257],
#     [1.0000000, 1.0000000, 0.0055520],
#     [2.0000000, 1.0000000, 0.0055520],
#     [3.0000000, 1.0000000, -0.0552257],
#     [0.0000000, 2.0000000, 0.0227398],
#     [1.0000000, 2.0000000, 0.0055520],
#     [2.0000000, 2.0000000, 0.0227398],
#     [3.0000000, 2.0000000, 0.0055520],
#     [0.0000000, 3.0000000, -0.0552257],
#     [1.0000000, 3.0000000, -0.0552257],
#     [2.0000000, 3.0000000, 0.0055520],
#     [3.0000000, 3.0000000, 0.0055520]
# ])

# sxy_corr = np.array([
#     [0.0000000, 0.0000000, 0.5000000],
#     [1.0000000, 0.0000000, -0.1184593],
#     [2.0000000, 0.0000000, 0.0767591],
#     [3.0000000, 0.0000000, -0.1184593],
#     [0.0000000, 1.0000000, -0.1184593],
#     [1.0000000, 1.0000000, -0.0014213],
#     [2.0000000, 1.0000000, -0.0014213],
#     [3.0000000, 1.0000000, -0.1184593],
#     [0.0000000, 2.0000000, 0.0767591],
#     [1.0000000, 2.0000000, -0.0014213],
#     [2.0000000, 2.0000000, 0.0767591],
#     [3.0000000, 2.0000000, -0.0014213],
#     [0.0000000, 3.0000000, -0.1184593],
#     [1.0000000, 3.0000000, -0.1184593],
#     [2.0000000, 3.0000000, -0.0014213],
#     [3.0000000, 3.0000000, -0.0014213]
# ])

# Nx = Ny = 4
# i = 0
# for ix in range(Nx):
#     for iy in range(Ny):
#         sf = 0
#         kx = 2 * np.pi * ix / Nx
#         ky = 2 * np.pi * iy / Ny
#         for row in sz_corr:
#             x, y, corr = row
#             k = np.array([kx, ky])
#             ri = np.array([0, 0])
#             rj = np.array([x, y])
#             sf += np.exp(-1j * k.dot(rj - ri)) * corr
#         print(sf)
