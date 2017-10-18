import numpy as np
from spinsys.hamiltonians.triangular_lattice_model import hamiltonian
from scipy import sparse
import spinsys


def structure_factor(vec, k, N):
    vec = sparse.csc_matrix(vec)
    obsv = []
    for bond in bonds:
        site1, site2 = bond
        i = site1.lattice_index
        j = site2.lattice_index
        coeff = np.exp(1j * (k.dot(site1 - site2)))
        obsv.append(coeff * z_mats[i].dot(z_mats[j]))
    return 1 / N * vec.T.conjugate().dot(sum(obsv)).dot(vec)[0, 0]


Nx = 4
Ny = 3
N = Nx * Ny
σz = spinsys.constructors.sigmaz()
z_mats = [spinsys.half.full_matrix(σz, k, N) for k in range(N)]

bonds = []
vec = spinsys.constructors.PeriodicBCSiteVector((0, 0), Nx, Ny)
for i in range(N):
    bonds.append((vec, vec.xhop(1)))
    bonds.append((vec, vec.yhop(1)))
    bonds.append((vec, vec.xhop(-1).yhop(1)))
    vec = vec.next_site()

H = hamiltonian(Nx, Ny)
E, ground_state = sparse.linalg.eigsh(H, k=1, which='SA')
for kx in np.linspace(0, 2 * np.pi, Nx, endpoint=False):
    for ky in np.linspace(0, 2 * np.pi, Ny, endpoint=False):
        fac = structure_factor(ground_state, np.array([kx, ky]), N)
        print((kx, ky), fac)
