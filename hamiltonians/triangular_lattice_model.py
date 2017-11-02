import numpy as np
from scipy import sparse
from spinsys import constructors, half, dmrg, utils


class SiteVector(constructors.PeriodicBCSiteVector):

    def __init__(self, ordered_pair, Nx, Ny):
        super().__init__(ordered_pair, Nx, Ny)

    def angle_with(self, some_site):
        """Returns the angle * 2 between (some_site - self) with the
        horizontal. Only works on nearest neighbors
        """
        Δx, Δy = some_site - self
        if Δx == 0:
            if Δy != 0:
                return -2 * np.pi / 3
        elif Δy == 0:
            if Δx != 0:
                return 0
        else:
            return 2 * np.pi / 3


def hamiltonian(Nx, Ny, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0):
    @utils.cache.cache_to_ram
    def pieces(Nx, Ny):
        """Generate the reusable pieces of the hamiltonian"""
        N = Nx * Ny

        σ_p = constructors.raising()
        σ_m = constructors.lowering()
        σz = constructors.sigmaz()

        p_mats = [half.full_matrix(σ_p, k, N) for k in range(N)]
        m_mats = [half.full_matrix(σ_m, k, N) for k in range(N)]
        z_mats = [half.full_matrix(σz, k, N) for k in range(N)]

        # Permute through all the nearest neighbor coupling bonds
        bonds = []
        vec = SiteVector((0, 0), Nx, Ny)
        for i in range(N):
            bonds.append((vec, vec.xhop(1)))
            bonds.append((vec, vec.yhop(1)))
            bonds.append((vec, vec.xhop(-1).yhop(1)))
            vec = vec.next_site()

        H_pm = H_z = H_ppmm = H_pmz = 0
        for bond in bonds:
            site1, site2 = bond
            i, j = site1.lattice_index, site2.lattice_index
            γ = np.exp(1j * site1.angle_with(site2))
            H_pm += p_mats[i].dot(m_mats[j]) + m_mats[i].dot(p_mats[j])
            H_z += z_mats[i].dot(z_mats[j])
            H_ppmm += γ * p_mats[i].dot(p_mats[j]) + \
                γ.conj() * m_mats[i].dot(m_mats[j])
            H_pmz += 1j * (γ.conj() * z_mats[i].dot(p_mats[j]) -
                           γ * z_mats[i].dot(m_mats[j]) +
                           γ.conj() * p_mats[i].dot(z_mats[j]) -
                           γ * m_mats[i].dot(z_mats[j]))
        return H_pm, H_z, H_ppmm, H_pmz

    H_pm, H_z, H_ppmm, H_pmz = pieces(Nx, Ny)
    return J_pm * H_pm + J_z * H_z + J_ppmm * H_ppmm + J_pmz * H_pmz


class DMRG_Hamiltonian(dmrg.Hamiltonian):

    def __init__(self):
        super().__init__()
        self.generators = {
            '+': constructors.raising(),
            '-': constructors.lowering(),
            'z': constructors.sigmaz()
        }

    def initialize_storage(self):
        init_block = sparse.csc_matrix(([], ([], [])), shape=[2, 2])
        init_ops = self.generators
        self.storage = dmrg.Storage(init_block, init_block, init_ops)

    def newsite_ops(self):
        pass

    def block_newsite_interaction(self):
        pass
