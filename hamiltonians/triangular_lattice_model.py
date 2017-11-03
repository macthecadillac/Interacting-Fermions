import copy
import numpy as np
from scipy import sparse
from spinsys import constructors, half, dmrg, utils, exceptions


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


class SemiPeriodicBCSiteVector(SiteVector):

    """A version of SiteVector that is periodic only along the x
    direction
    """

    def __init__(self, ordered_pair, Nx, Ny):
        super().__init__(ordered_pair, Nx, Ny)

    def diff(self, other):
        """Finds the shortest distance from this site to the other"""
        Δx = self.x - other.x
        Δy = self.y - other.y
        return (Δx, Δy)

    def yhop(self, stride):
        new_vec = copy.copy(self)
        new_y = self.y + stride
        if new_y // self.Ny == self.x // self.Ny:
            new_vec.y = new_y
        else:
            raise exceptions.OutOfBoundsError("Hopping off the lattice")
        return new_vec

    @property
    def neighboring_sites(self):
        neighbors = []
        funcs = [self.xhop, self.yhop]
        for Δ in [1, -1]:
            for func in funcs:
                try:
                    neighbors.append(func(Δ).lattice_index)
                except exceptions.OutOfBoundsError:
                    continue
            try:
                neighbors.append(self.xhop(Δ).yhop(-Δ).lattice_index)
            except exceptions.OutOfBoundsError:
                continue
        return neighbors


def hamiltonian(Nx, Ny, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0):
    """Generates hamiltonian for the triangular lattice model. No
    optimizations built in.

    Parameters
    --------------------
    Nx: int
        number of sites along the x-direction
    Ny: int
        number of sites along the y-direction
    J_pm: int
        the J_+- parameter
    J_z: int
        the J_z parameter
    J_ppmm: int
        the J_++-- parameter
    J_pmz: int
        the J_+-z parameter

    Returns
    --------------------
    H: scipy.sparse.csc_matrix
    """
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

            H_ppmm += \
                γ * p_mats[i].dot(p_mats[j]) + \
                γ.conj() * m_mats[i].dot(m_mats[j])

            H_pmz += 1j * (γ.conj() * z_mats[i].dot(p_mats[j]) -
                           γ * z_mats[i].dot(m_mats[j]) +
                           γ.conj() * p_mats[i].dot(z_mats[j]) -
                           γ * m_mats[i].dot(z_mats[j]))

        return H_pm, H_z, H_ppmm, H_pmz

    H_pm, H_z, H_ppmm, H_pmz = pieces(Nx, Ny)
    return J_pm * H_pm + J_z * H_z + J_ppmm * H_ppmm + J_pmz * H_pmz


def hamiltonian_Tx():
    pass


class DMRG_Hamiltonian(dmrg.Hamiltonian):

    def __init__(self, Nx, Ny, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0):
        self.generators = {
            '+': constructors.raising(),
            '-': constructors.lowering(),
            'z': constructors.sigmaz()
        }
        self.N = Nx * Ny
        self.Nx = Nx
        self.Ny = Ny
        self.J_pm = J_pm
        self.J_z = J_z
        self.J_ppmm = J_ppmm
        self.J_pmz = J_pmz
        super().__init__()

    def initialize_storage(self):
        init_block = sparse.csc_matrix(([], ([], [])), shape=[2, 2])
        init_ops = self.generators
        self.storage = dmrg.Storage(init_block, init_block, init_ops)

    def newsite_ops(self, size):
        return dict((i, sparse.kron(sparse.eye(size // 2), self.generators[i]))
                    for i in self.generators.keys())

    # TODO: Inconsistent shapes error at runtime
    def block_newsite_interaction(self, block_key):
        block_side, curr_site = block_key
        site = SemiPeriodicBCSiteVector.from_index(curr_site, self.Nx, self.Ny)
        neighbors = [i for i in site.neighboring_sites if i < curr_site]

        H_pm_new = H_z_new = H_ppmm_new = H_pmz_new = sparse.csc_matrix(np.zeros((2, 2)))
        for i in neighbors:
            key = (block_side, i + 1)
            block_ops = self.storage.get_item(key).ops
            site_ops = self.generators

            H_pm_new += \
                sparse.kron(block_ops['+'], site_ops['-']) + \
                sparse.kron(block_ops['-'], site_ops['+'])

            H_z_new += sparse.kron(block_ops['z'], site_ops['z'])

            H_ppmm_new += \
                sparse.kron(block_ops['+'], site_ops['+']) + \
                sparse.kron(block_ops['-'], site_ops['-'])

            H_pmz_new += \
                sparse.kron(block_ops['z'], site_ops['+']) + \
                sparse.kron(block_ops['z'], site_ops['-']) + \
                sparse.kron(block_ops['+'], site_ops['z']) + \
                sparse.kron(block_ops['-'], site_ops['z'])

        return self.J_pm * H_pm_new + self.J_z * H_z_new + \
            self.J_ppmm * H_ppmm_new + self.J_pmz * H_pmz_new
