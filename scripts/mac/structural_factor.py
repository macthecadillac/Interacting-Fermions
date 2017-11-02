import numpy as np
from scipy import sparse
from spinsys import constructors, half, utils
import functools
from hamiltonians.triangular_lattice_model import hamiltonian, SiteVector


def structural_factor_v1(N, kx, ky, ψ0):
    @functools.lru_cache(maxsize=None)
    def create_z_mats(N):
        σz = constructors.sigmaz()
        return [half.full_matrix(σz, k, N) for k in range(N)]

    @functools.lru_cache(maxsize=None)
    def gen_site_lists(Nx, Ny):
        sites = []
        vec = SiteVector((0, 0), Nx, Ny)
        for i in range(N):
            sites.append(vec.coord)
            vec = vec.next_site()
        return sites

    z_mats = create_z_mats(N)
    sites = gen_site_lists(Nx, Ny)
    k = np.array([kx, ky])
    ftrans_factors = np.array([np.exp(-1j * k.dot(vec)) for vec in sites])
    ftransformed_σz = ftrans_factors.dot(z_mats)
    ftransformed_SS = ftransformed_σz.conj().dot(ftransformed_σz)
    return 1 / N * ψ0.T.conj().dot(ftransformed_SS).dot(ψ0)[0, 0]


def structural_factor_v2(N, kx, ky, ψ0):
    @functools.lru_cache(maxsize=None)
    def spin_correlation_vals(N):
        σz = constructors.sigmaz()
        z_mats = [half.full_matrix(σz, k, N) for k in range(N)]
        Sij = []
        for i in range(N):
            for j in range(i + 1, N):
                Sij.append(z_mats[i].dot(z_mats[j]))
        return np.array([ψ0.T.conj().dot(S).dot(ψ0) for S in Sij])

    @functools.lru_cache(maxsize=None)
    def gen_site_lists(Nx, Ny):
        sites = []
        vec = SiteVector((0, 0), Nx, Ny)
        for i in range(N):
            sites.append(vec.coord)
            vec = vec.next_site()
        return sites

    sites = gen_site_lists(Nx, Ny)
    k = np.array([kx, ky])
    ftrans_factors = np.array([np.exp(-1j * k.dot(vec)) for vec in sites])
    return ftrans_factors.dot(spin_correlation_vals(N))


Nx = 4
Ny = 3
N = Nx * Ny
H = hamiltonian(Nx, Ny)
E, ψ0 = sparse.linalg.eigsh(H, k=1, which='SA')
ψ0 = sparse.csc_matrix(ψ0)
nkx = nky = 100
data = np.empty((nkx, nky), dtype=complex)
t = utils.timer.Timer(nkx * nky)
for i, kx in enumerate(np.linspace(-np.pi, np.pi, nkx, endpoint=False)):
    for j, ky in enumerate(np.linspace(-np.pi, np.pi, nky, endpoint=False)):
        s = structural_factor_v1(N, kx, ky, ψ0)
        data[i, j] = s
        t.progress()
np.save('./structural_factor.npy', data, allow_pickle=False)
