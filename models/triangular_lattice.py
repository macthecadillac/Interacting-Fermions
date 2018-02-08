import copy
import functools
import os
import numpy as np
from scipy import sparse
from spinsys import constructors, half, dmrg, exceptions
from cffi import FFI


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

    def a1_hop(self, stride):
        vec = self.xhop(stride)
        if vec == self:
            raise exceptions.SameSite
        return vec

    def a2_hop(self, stride):
        vec = self.xhop(-1 * stride).yhop(stride)
        if vec == self:
            raise exceptions.SameSite
        return vec

    def a3_hop(self, stride):
        vec = self.yhop(-stride)
        if vec == self:
            raise exceptions.SameSite
        return vec

    def b1_hop(self, stride):
        """hop in the a1 - a3 aka b1 direction. Useful for second nearest
        neighbor coupling interactions
        """
        vec = self.xhop(stride).yhop(stride)
        if vec == self:
            raise exceptions.SameSite
        return vec

    def b2_hop(self, stride):
        vec = self.xhop(-2 * stride).yhop(stride)
        if vec == self:
            raise exceptions.SameSite
        return vec

    def b3_hop(self, stride):
        vec = self.b1_hop(-stride).b2_hop(-stride)
        if vec == self:
            raise exceptions.SameSite
        return vec

    def _neighboring_sites(self, strides, funcs):
        neighbors = []
        for stride in strides:
            for func in funcs:
                try:
                    neighbors.append(func(stride))
                except exceptions.SameSite:
                    continue
        return neighbors

    @property
    def nearest_neighboring_sites(self, all=False):
        strides = [1, -1] if all else [1]
        funcs = [self.a1_hop, self.a2_hop, self.a3_hop]
        return self._neighboring_sites(strides, funcs)

    @property
    def second_neighboring_sites(self, all=False):
        """with the all option enabled the method will enumerate all
        the sites that are second neighbors to the current site.
        Otherwise it will only enumerate the sites along the b1, b2
        and b3 directions
        """
        strides = [1, -1] if all else [1]
        funcs = [self.b1_hop, self.b2_hop, self.b3_hop]
        return self._neighboring_sites(strides, funcs)

    @property
    def third_neighboring_sites(self, all=False):
        strides = [2, -2] if all else [2]
        funcs = [self.a1_hop, self.a2_hop, self.a3_hop]
        return self._neighboring_sites(strides, funcs)


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


@functools.lru_cache(maxsize=None)
def _generate_bonds(Nx, Ny):
    N = Nx * Ny
    vec = SiteVector((0, 0), Nx, Ny)
    # range_orders = [set(), set(), set()]  # sets de-duplicates the list of bonds
    range_orders = [[], [], []]
    for i in range(N):
        nearest_neighbor = vec.nearest_neighboring_sites
        second_neighbor = vec.second_neighboring_sites
        third_neighbor = vec.third_neighboring_sites
        neighbors = [nearest_neighbor, second_neighbor, third_neighbor]
        for leap, bonds in enumerate(range_orders):
            for n in neighbors[leap]:
                # sort them so identical bonds will always have the same hash
                bond = sorted((vec, n))
                bonds.append(tuple(bond))
        vec = vec.next_site()
    return range_orders


@functools.lru_cache(maxsize=None)
def _gen_full_ops(N):
    σ_p = constructors.raising()
    σ_m = constructors.lowering()
    σz = constructors.sigmaz()
    p_mats = [half.full_matrix(σ_p, k, N) for k in range(N)]
    m_mats = [half.full_matrix(σ_m, k, N) for k in range(N)]
    z_mats = [half.full_matrix(σz, k, N) for k in range(N)]
    return p_mats, m_mats, z_mats


def _gen_z_pm_ops(N, bonds):
    """generate the H_z and H_pm components of the Hamiltonian"""
    H_pm = H_z = 0
    p_mats, m_mats, z_mats = _gen_full_ops(N)
    for bond in bonds:
        site1, site2 = bond
        i, j = site1.lattice_index, site2.lattice_index
        H_pm += p_mats[i].dot(m_mats[j]) + m_mats[i].dot(p_mats[j])
        H_z += z_mats[i].dot(z_mats[j])
    return H_pm, H_z


@functools.lru_cache(maxsize=None)
def hamiltonian_dp_components(Nx, Ny):
    """Generate the reusable pieces of the hamiltonian"""
    N = Nx * Ny
    nearest, second, third = _generate_bonds(Nx, Ny)
    H_pm1, H_z1 = _gen_z_pm_ops(N, nearest)
    H_pm2, H_z2 = _gen_z_pm_ops(N, second)
    H_pm3, H_z3 = _gen_z_pm_ops(N, third)

    H_ppmm = H_pmz = 0
    p_mats, m_mats, z_mats = _gen_full_ops(N)
    for bond in nearest:
        site1, site2 = bond
        i, j = site1.lattice_index, site2.lattice_index
        γ = np.exp(1j * site1.angle_with(site2))

        H_ppmm += \
            γ * p_mats[i].dot(p_mats[j]) + \
            γ.conj() * m_mats[i].dot(m_mats[j])

        H_pmz += 1j * (γ.conj() * z_mats[i].dot(p_mats[j]) -
                       γ * z_mats[i].dot(m_mats[j]) +
                       γ.conj() * p_mats[i].dot(z_mats[j]) -
                       γ * m_mats[i].dot(z_mats[j]))

    return H_pm1, H_z1, H_ppmm, H_pmz, H_pm2, H_z2, H_z3, H_pm3


def hamiltonian_dp(Nx, Ny, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0, J2=0, J3=0):
    """Generates hamiltonian for the triangular lattice model with
    direct product

    Parameters
    --------------------
    Nx: int
        number of sites along the x-direction
    Ny: int
        number of sites along the y-direction
    J_pm: float
        J_+- parameter
    J_z: float
        J_z parameter
    J_ppmm: float
        J_++-- parameter
    J_pmz: float
        J_+-z parameter
    J2: float
        second nearest neighbor interaction parameter
    J3: float
        third nearest neighbor interaction parameter

    Returns
    --------------------
    H: scipy.sparse.csc_matrix
    """
    components = hamiltonian_dp_components(Nx, Ny)
    H_pm1, H_z1, H_ppmm, H_pmz, H_pm2, H_z2, H_z3, H_pm3 = components
    nearest_neighbor_terms = J_pm * H_pm1 + J_z * H_z1 + J_ppmm * H_ppmm + J_pmz * H_pmz
    second_neighbor_terms = third_neighbor_terms = 0
    if not J2 == 0:
        second_neighbor_terms = J2 * (H_pm2 + J_z / J_pm * H_z2)
    if not J3 == 0:
        third_neighbor_terms = J3 * (H_pm3 + J_z / J_pm * H_z3)
    return nearest_neighbor_terms + second_neighbor_terms + third_neighbor_terms


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
        init_block = sparse.csc_matrix(([], ([], [])), dims=[2, 2])
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

        H_pm_new = H_z_new = H_ppmm_new = H_pmz_new = 0
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


ffi = FFI()
header = """
typedef struct {
    double * ptr;
    size_t len;
} vector;


typedef struct {
    vector data;
    vector col;
    vector row;
    unsigned int ncols;
    unsigned int nrows;
} coordmatrix;


coordmatrix hamiltonian(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        double,
        double,
        double,
        double,
        double,
        double
);

coordmatrix ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ss_pm(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

void request_free(coordmatrix);
"""
ffi.cdef(header)
_lib = ffi.dlopen(os.path.join(os.path.dirname(__file__),
                               "triangular_lattice_ext.so"))


class CoordMatrix:
    """A class that encapsulates the matrix and provides methods that would
    help memoery management over the FFI
    """

    def __init__(self, mat):
        """Initializer

        Parameters
        --------------------
        mat: CoordMatrix
        """
        self.__obj = mat  # the pointer to the pointers to the arrays
        self.data = np.frombuffer(ffi.buffer(mat.data.ptr, mat.data.len * 16),
                                  np.complex128)
        self.col = np.frombuffer(ffi.buffer(mat.col.ptr, mat.col.len * 4),
                                 np.int32)
        self.row = np.frombuffer(ffi.buffer(mat.row.ptr, mat.row.len * 4),
                                 np.int32)
        self.ncols = mat.ncols
        self.nrows = mat.nrows

    def __enter__(self):
        """For use with context manager"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """For use with context manager"""
        self.data = None
        self.col = None
        self.row = None
        _lib.request_free(self.__obj)
        self.__obj = None

    def to_csc(self):
        """Returns a CSC matrix"""
        return sparse.csc_matrix((self.data, (self.col, self.row)),
                                 shape=(self.nrows, self.ncols))

    def to_csr(self):
        """Returns a CSR matrix"""
        return sparse.csr_matrix((self.data, (self.col, self.row)),
                                 shape=(self.nrows, self.ncols))


def hamiltonian_consv_k(Nx, Ny, kx, ky, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0, J2=0, J3=0):
    """construct the full Hamiltonian matrix in the given momentum configuration

    Parameters
    --------------------
    Nx: int
        lattice length in the x-direction
    Ny: int
        lattice length in the y-direction
    kx: int
        the x-component of lattice momentum * Nx / 2π in a (-π, +π]
        Brillouin zone
    ky: int
        the y-component of lattice momentum * Ny / 2π in a (-π, +π]
        Brillouin zone
    J_pm: int/float
        the J+- parameter (defaults to 0)
    J_z: int/float
        the Jz parameter (defaults to 0)
    J_ppmm: int/float
        the J++-- parameter (defaults to 0)
    J_pmz: int/float
        the J+-z parameter (defaults to 0)
    J2: int/float
        the J2 parameter (defaults to 0)
    J3: int/float
        the J3 parameter (defaults to 0)

    Returns
    --------------------
    H: CoordMatrix (See above)
    """

    mat = _lib.hamiltonian(Nx, Ny, kx, ky, J_z, J_pm, J_ppmm, J_pmz, J2, J3)
    coordmat = CoordMatrix(mat)
    return coordmat


def ss_z(Nx, Ny, kx, ky, l):
    """construct the Σsz_i * sz_j operators with the given separation
    with translational symmetry taken into account

    Parameters
    --------------------
    Nx: int
        lattice length in the x-direction
    Ny: int
        lattice length in the y-direction
    kx: int
        the x-component of lattice momentum * Nx / 2π in a (-π, +π]
        Brillouin zone
    ky: int
        the y-component of lattice momentum * Ny / 2π in a (-π, +π]
        Brillouin zone
    l:  int
        the separation between sites: |i - j|

    Returns
    --------------------
    ss_z: CoordMatrix (See above)
    """
    mat = _lib.ss_z(Nx, Ny, kx, ky, l)
    coordmat = CoordMatrix(mat)
    return coordmat


def ss_pm(Nx, Ny, kx, ky, l):
    """construct the Σsz_i * sz_j operators with the given separation
    with translational symmetry taken into account

    Parameters
    --------------------
    Nx: int
        lattice length in the x-direction
    Ny: int
        lattice length in the y-direction
    kx: int
        the x-component of lattice momentum * Nx / 2π in a (-π, +π]
        Brillouin zone
    ky: int
        the y-component of lattice momentum * Ny / 2π in a (-π, +π]
        Brillouin zone
    l:  int
        the separation between sites: |i - j|

    Returns
    --------------------
    ss_z: CoordMatrix (See above)
    """
    mat = _lib.ss_pm(Nx, Ny, kx, ky, l)
    coordmat = CoordMatrix(mat)
    return coordmat


def min_necessary_ks(Nx, Ny):
    """Returns the momentum that we absolutely need to compute

    Parameters
    --------------------
    Nx: int
    Ny: int

    Returns
    --------------------
    list of ints
    """
    ks = []
    arrs = []
    for kx in range(Nx):
        for ky in range(Ny):
            arr = np.outer(np.exp(2j * np.pi * kx * np.arange(Nx) / Nx),
                           np.exp(2j * np.pi * ky * np.arange(Ny) / Ny))
            for arr0 in arrs:
                if np.allclose(arr0, arr) or np.allclose(arr0, arr.conjugate()):
                    break
            else:
                ks.append((kx, ky))
                arrs.append(arr)
    return ks
