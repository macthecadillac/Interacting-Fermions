import numpy as np
from scipy import sparse
from spinsys import constructors, half, utils
from hamiltonians.triangular_lattice_model import hamiltonian, SiteVector


def structural_factor(N, kx, ky, ψ0):
    σz = constructors.sigmaz()
    z_mats = [half.full_matrix(σz, k, N) for k in range(N)]
    sites = []
    vec = SiteVector((0, 0), Nx, Ny)
    for i in range(N):
        sites.append(vec.coord)
        vec = vec.next_site()

    k = np.array([kx, ky])
    ftrans_factors = np.array([np.exp(-1j * k.dot(vec)) for vec in sites])
    ftransformed_σz = ftrans_factors.dot(z_mats)
    ftransformed_SS = ftransformed_σz.conj().dot(ftransformed_σz)
    return 1 / N * ψ0.T.conj().dot(ftransformed_SS).dot(ψ0)[0, 0]


Nx = 4
Ny = 3
N = Nx * Ny
H = hamiltonian(Nx, Ny)
E, ψ0 = sparse.linalg.eigsh(H, k=1, which='SA')
ψ0 = sparse.csc_matrix(ψ0)
nkx = nky = 1000
data = np.empty((nkx, nky), dtype=complex)
t = utils.timer.Timer(nkx * nky)
for i, kx in enumerate(np.linspace(-np.pi, np.pi, nkx, endpoint=False)):
    for j, ky in enumerate(np.linspace(-np.pi, np.pi, nky, endpoint=False)):
        s = structural_factor(N, kx, ky, ψ0)
        data[i, j] = s
        t.progress()
np.savetxt('./structural_factor.txt', data)
