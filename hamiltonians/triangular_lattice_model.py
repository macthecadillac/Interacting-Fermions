from collections import namedtuple
import copy
import functools
import numpy as np
from scipy import sparse
from spinsys import constructors, half, dmrg, exceptions


Pair = namedtuple('Pair', 'x, y')
Momentum = namedtuple('Momentum', 'kx, ky')
BlochFunc = namedtuple('BlochFunc', 'lead, decs, dims')
BlochFunc.__hash__ = lambda inst: hash((inst.lead, inst.dims))


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
    range_orders = [set(), set(), set()]  # sets de-duplicates the list of bonds
    for i in range(N):
        nearest_neighbor = vec.nearest_neighboring_sites
        second_neighbor = vec.second_neighboring_sites
        third_neighbor = vec.third_neighboring_sites
        neighbors = [nearest_neighbor, second_neighbor, third_neighbor]
        for leap, bonds in enumerate(range_orders):
            for n in neighbors[leap]:
                # sort them so identical bonds will always have the same hash
                bond = sorted((vec, n))
                bonds.add(tuple(bond))
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
    H_pm1, H_z1, H_ppmm, H_pmz, H_pm2, H_z2, H_pm3, H_z3 = components
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


class BlochTable:
    """A complex custom datatype.

    Let's say "states" is a BlochTable, then it has the following
    fields/attributes:

    states.leading_states: list

        list of leading states in decimal representation

    states.bloch_state: dict

        a dictionary that maps any product state in the Hilbert space
        to the full Bloch state it is a part of

    BlochTable is also iterable and could be indexed.

    states[i]: BlochFunc
        BlochFunc is another datatype defined above and has the following
        fields/attributes:

        states[i].lead: int

            the leading product state of a Bloch state

        states[i].decs: numpy.array

            the full Bloch state in a 2-D array. Traversal along the array
            rows amounts to translation in the x-direction on the lattice and
            traversal along the array columns amouns to translation in the
            y-direction on the lattice. The 0th element is the leading state.

        states[i].dims: Pair

            the "dimensions", for lack of a better word, of a Bloch state.

            states[i].dims.x: int
                smallest integer n such that T^n = 1 on a any product
                state that is part of this Bloch state along the x-direction

            states[i].dims.y: int
                smallest integer n such that T^n = 1 on a any product
                state that is part of this Bloch state along the y-direction
    """

    def __init__(self, states, shapes):
        leading_states = [s[0, 0] for s in states]
        self.data = list(zip(leading_states, states, shapes))
        self.bloch_state = {}
        self._populate_dict()

    def __getitem__(self, i):
        return BlochFunc(*self.data[i])

    def __len__(self):
        return len(self.data)

    def _populate_dict(self):
        for state in self.data:
            state_pkg = BlochFunc(*state)
            for dec in state[1].flatten():
                self.bloch_state[dec] = state_pkg

    def sort(self):
        table = sorted(self.data, key=lambda x: x[0])
        self.data = table

    @property
    def leading_states(self):
        return tuple(zip(*self.data))[0]


@functools.lru_cache(maxsize=None)
def _roll_x_aux(Nx, Ny):
    n = np.arange(0, Nx * Ny, Nx)
    a = 2 ** (n + Nx)
    b = 2 ** n
    c = 2 ** Nx
    d = 2 ** (Nx - 1)
    e = 2 ** n
    return a, b, c, d, e


@functools.lru_cache(maxsize=None)
def roll_x(dec, Nx, Ny):
    """translates a given state along the x-direction for one site.
    assumes periodic boundary condition.

    Parameters
    --------------------
    dec: int
        the decimal representation of a product state.
    Nx: int
        lattice size along the x-direction
    Ny: int
        lattice size along the y-direction

    Returns
    --------------------
    state': int
        the new state after translation
    """
    a, b, c, d, e = _roll_x_aux(Nx, Ny)
    s = dec % a // b    # "%" is modulus and "//" is integer division
    s = (s * 2) % c + s // d
    return (e).dot(s)


@functools.lru_cache(maxsize=None)
def _roll_y_aux(Nx, Ny):
    return 2 ** Nx, 2 ** (Nx * (Ny - 1))


# cannot be memoized because state is a numpy array which is not hashable
def roll_y(arr, Nx, Ny):
    """translates a given state along the y-direction for one site.
    assumes periodic boundary condition.

    Parameters
    --------------------
    arr: numpy.array / int
        the decimal representation of a product state.
    Nx: int
        lattice size along the x-direction
    Ny: int
        lattice size along the y-direction

    Returns
    --------------------
    arr': int
        the new state after translation
    """
    xdim, pred_totdim = _roll_y_aux(Nx, Ny)
    tail = arr % xdim
    return arr // xdim + tail * pred_totdim


@functools.lru_cache(maxsize=None)
def _exchange_spin_flips(dec, b1, b2):
    """tests whether a given state constains a spin flip at sites
    represented by b1 and b2.

    Parameters
    --------------------
    dec: int
        the decimal representation of a product state.
    b1: int
        the decimal representation of bit 1 to be examined
    b2: int
        the decimal representation of bit 2 to be examined

    Returns
    --------------------
    updown: bool
    downup: bool
    """
    updown = downup = False
    if (dec | b1 == dec) and (not dec | b2 == dec):
        updown = True
    if (not dec | b1 == dec) and (dec | b2 == dec):
        downup = True
    return updown, downup


@functools.lru_cache(maxsize=None)
def _repeated_spins(dec, b1, b2):
    """tests whether both spins at b1 and b2 point in the same direction.

    Parameters
    --------------------
    dec: int
        the decimal representation of a product state.
    b1: int
        the decimal representation of bit 1 to be examined
    b2: int
        the decimal representation of bit 2 to be examined

    Returns
    --------------------
    upup: bool
    downdown: bool
    """
    upup = downdown = False
    if (dec | b1 == dec) and (dec | b2 == dec):
        upup = True
    if (not dec | b1 == dec) and (not dec | b2 == dec):
        downdown = True
    return upup, downdown


@functools.lru_cache(maxsize=None)
def _unnomalized_phases(N, k, length):
    """generates the un-normalized coefficients of individual product
    states within a Bloch state.

    Parameters
    --------------------
    N: int
        lattice length in one unspecified direction
    k: int
        lattice momentum * N / 2π in a (-π, +π] Brillouin zone
    length: int
        smallest integer n such that T^n = 1 on a specific product state
        along the aforementioned unspecified direction

    Returns
    --------------------
    coeffs: numpy.array
        coefficients of individual product states within a Bloch state.
    """
    mprime_max = N // length
    m = np.arange(length).repeat(mprime_max).reshape(-1, mprime_max).T
    mprime = np.arange(mprime_max).repeat(length).reshape(mprime_max, -1)
    phases = np.exp(2j * np.pi * k * (mprime * length + m) / N)
    coeffs = np.sum(phases, axis=0)
    return coeffs


@functools.lru_cache(maxsize=None)
def _phase_arr(Nx, Ny, kx, ky, dims):
    coeffs_x = _unnomalized_phases(Nx, kx, dims.x)
    coeffs_y = _unnomalized_phases(Ny, ky, dims.y)
    # outer product of the coefficients along the x and y directions
    return np.outer(coeffs_y, coeffs_x)


@functools.lru_cache(maxsize=None)
def _norm_coeff(Nx, Ny, kx, ky, dims):
    """generates the norm of a given configuration, akin to the reciprocal
    of the normalization factor.

    Paramters
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
    dims: Pair
        The x field contains the smallest integer nx such that T^nx = 1
        on a specific product state along the x-direction. Same goes the
        y field.

    Returns
    --------------------
    coeff: float
        the normalization factor
    """
    return np.linalg.norm(_phase_arr(Nx, Ny, kx, ky, dims))


@functools.lru_cache(maxsize=None)
def _gamma(Nx, Ny, b1, b2):
    """calculates γ"""
    m = int(round(np.log2(b1)))
    n = int(round(np.log2(b2)))
    vec1 = SiteVector.from_index(m, Nx, Ny)
    vec2 = SiteVector.from_index(n, Nx, Ny)
    ang = vec1.angle_with(vec2)
    return np.exp(1j * ang)


@functools.lru_cache(maxsize=None)
def _bits(Nx, Ny, l):
    """generates the integers that represent interacting sites

    Parameters
    --------------------
    Nx: int
        lattice length in the x-direction
    Ny: int
        lattice length in the y-direction
    l: int
        range of interaction. 1 for nearest neighbor interation, 2 for
        second neighbors, 3 for third neighbors

    Returns
    --------------------
    bit1: numpy.array
        array of integers that are decimal representations of single sites
    bit2: numpy.array
        array of integers that are decimal representations of sites that
        when taken together (zip) with bit1 locates interacting sites
    """
    bit1, bit2 = [], []
    bond_orders = _generate_bonds(Nx, Ny)
    bonds = bond_orders[l - 1]
    for bond in bonds:
        bit1.append(bond[0].lattice_index)
        bit2.append(bond[1].lattice_index)
    bit1 = np.array(bit1)
    bit2 = np.array(bit2)
    return 2 ** bit1, 2 ** bit2


@functools.lru_cache(maxsize=None)
def zero_momentum_states(Nx, Ny):
    """finds a full set of bloch states with zero lattice momentum

    Parameters
    --------------------
    Nx: int
        lattice length in the x-direction
    Ny: int
        lattice length in the y-direction

    Returns
    --------------------
    states: dict
        a dictionary that maps "dimensions" to Bloch states
    """
    def find_T_invariant_set(i):
        """takes an integer "i" as the leading state and translates it
        repeatedly along the x and y-directions until we find the entire
        set
        """
        b = [i]
        j = i
        while True:
            j = roll_x(j, Nx, Ny)
            if not j == i:
                b.append(j)
                sieve[j] = 0
            else:
                break
        b = np.array(b)
        bloch_func = [b]
        u = b.copy()
        while True:
            u = roll_y(u, Nx, Ny)
            if not np.sort(u)[0] == np.sort(b)[0]:
                bloch_func.append(u)
                sieve[u] = 0
            else:
                break
        bloch_func = np.array(bloch_func, dtype=np.int64)
        try:
            dims = bloch_func.shape
            s = Pair(x=dims[1], y=dims[0])
            states[s].append(bloch_func)
        except KeyError:
            states[s] = [bloch_func]

    N = Nx * Ny
    sieve = np.ones(2 ** N, dtype=np.int8)
    sieve[0] = sieve[-1] = 0
    maxdec = 2 ** N - 1
    states = {Pair(1, 1): [np.array([[0]]), np.array([[maxdec]])]}
    for i in range(maxdec + 1):
        if sieve[i]:
            find_T_invariant_set(i)
    return states


@functools.lru_cache(maxsize=None)
def _bloch_states(Nx, Ny, kx, ky):
    """picks out the Bloch states from the zero-momentum set that
    are non-zero in the given momentum

    Paramters
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

    Returns
    --------------------
    table: BlochTable
    """
    def check_bounds(N, k):
        return k < 0 and (-k == N // 2 and N % 2 == 0)

    # I'm restricting myself to within the first Brillouin zone here
    # and invalidates anything that falls outside of it
    if abs(kx) > Nx // 2 or abs(ky) > Ny // 2:
        raise exceptions.NotFoundError
    elif check_bounds(Nx, kx) or check_bounds(Ny, ky):
        raise exceptions.NotFoundError

    zero_k_states = zero_momentum_states(Nx, Ny)
    states, shapes = [], []
    for dims, state_list in zero_k_states.items():
        # this computes the norm of such states under some arbitrary
        # momentum configuration and see if it is non-zero
        norm_coeff = _norm_coeff(Nx, Ny, kx, ky, dims)
        if norm_coeff > 1e-8:
            nstates = len(state_list)
            # store the state and associated metadata if its norm
            # is non-zero
            states.extend(state_list)
            shapes.extend([dims] * nstates)

    if not states:
        raise exceptions.NotFoundError

    table = BlochTable(states, shapes)
    table.sort()
    return table


@functools.lru_cache(maxsize=None)
def _find_leading_state(Nx, Ny, kx, ky, dec):
    """finds the leading state for a given state

    Paramters
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
    dec: int
        the decimal representation of a product state.

    Returns
    --------------------
    lead: int
        the decimal representation of the leading state
    phase: float
        the phase associated with the given state
    """
    bloch_states = _bloch_states(Nx, Ny, kx, ky)

    try:
        cntd_state = bloch_states.bloch_state[dec]
    except KeyError:
        raise exceptions.NotFoundError

    # trace how far the given state is from the leading state by translation
    iy, ix = np.where(cntd_state.decs == dec)
    ix, iy = ix[0], iy[0]  # ix and iy are single element arrays. we want floats
    xphase = np.exp(2j * np.pi * kx * ix / Nx)
    yphase = np.exp(2j * np.pi * ky * iy / Ny)
    phase = xphase * yphase
    return cntd_state, phase


@functools.lru_cache(maxsize=None)
def _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky):
    states = _bloch_states(Nx, Ny, kx, ky)
    dec = states.leading_states
    nstates = len(dec)
    inds = list(range(nstates))
    dec_to_ind = dict(zip(dec, inds))
    ind_to_dec = dict(zip(inds, states))
    return ind_to_dec, dec_to_ind


@functools.lru_cache(maxsize=None)
def _coeff(Nx, Ny, kx, ky, i, j):
    """calculates the coefficient that connects matrix indices i and j
    (sans phase)

    Paramters
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
    i: int
        index of the Bloch state before the Hamiltonian acts on it
    j: int
        index of the Bloch state the Hamiltonian connects state i with

    Returns
    --------------------
    coeff: float
    """
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    orig_state = ind_to_dec[i]
    cntd_state = ind_to_dec[j]
    # normalization factors
    normfac_i = _norm_coeff(Nx, Ny, kx, ky, orig_state.dims)
    normfac_j = _norm_coeff(Nx, Ny, kx, ky, cntd_state.dims)
    coeff = normfac_j / normfac_i
    return coeff


def H_z_elements(Nx, Ny, kx, ky, i, l):
    """computes the Hz elements

    Paramters
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
    i: int
        index of the Bloch state before the Hamiltonian acts on it
    l: int
        range of interaction. 1 for nearest neighbor interation, 2 for
        second neighbors, 3 for third neighbors

    Returns
    --------------------
    H_ii: float
        the i'th element of Hz
    """
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i]
    bit1, bit2 = _bits(Nx, Ny, l)
    same_dir = 0
    # b1 is the decimal representation of a spin-up at site 1 and
    # b2 is the decimal representation of a spin-up at site 2
    for b1, b2 in zip(bit1, bit2):
        upup, downdown = _repeated_spins(state.lead, b1, b2)
        same_dir += upup + downdown
    diff_dir = len(bit1) - same_dir
    return 0.25 * (same_dir - diff_dir)


@functools.lru_cache(maxsize=None)
def _zero_coeff(state, Nx, Ny, kx, ky):
    coeff = 1
    phase_arr = _phase_arr(Nx, Ny, kx, ky, state.dims)
    row1 = roll_y(state.decs[-1, :], Nx, Ny)
    row2 = state.decs[0, :]
    col1 = np.array([roll_x(i, Nx, Ny) for i in state.decs[:, -1]])
    col2 = state.decs[:, 0]
    if ky != 0 and state.dims.x < Nx:
        if not np.array_equal(col1, col2):
            offset = np.where(col2 == col1[0])[0][0]
            if abs(phase_arr[0, 0] - phase_arr[offset, 0]) > 1e-8:
                coeff = 0
    if kx != 0 and state.dims.y < Ny:
        if not np.array_equal(row1, row2):
            offset = np.where(row2 == row1[0])[0][0]
            if abs(phase_arr[0, 0] - phase_arr[0, offset]) > 1e-8:
                coeff = 0
    return coeff


def H_pm_elements(Nx, Ny, kx, ky, i, l):
    """computes the H+- elements

    Paramters
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
    i: int
        index of the Bloch state before the Hamiltonian acts on it
    l: int
        range of interaction. 1 for nearest neighbor interation, 2 for
        second neighbors, 3 for third neighbors

    Returns
    --------------------
    j_element: dict
        a dictionary that maps j's to their values for a given i
    """
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i]
    j_element = {}
    bits = _bits(Nx, Ny, l)
    # b1 is the decimal representation of a spin-up at site 1 and
    # b2 is the decimal representation of a spin-up at site 2
    for b1, b2 in zip(*bits):
        # updown and downup are booleans
        updown, downup = _exchange_spin_flips(state.lead, b1, b2)
        if updown or downup:
            # if the configuration is updown, we flip the spins by
            # turning the spin-up to spin-down and vice versa
            if updown:  # if updown == True
                new_state = state.lead - b1 + b2
            elif downup:  # if downup == True
                new_state = state.lead + b1 - b2

            try:
                # find what connected state it is if the state we got from bit-
                #  flipping is not in our records
                cntd_state, phase = _find_leading_state(Nx, Ny, kx, ky, new_state)
                # once we have the leading state, we proceed to find the
                # corresponding matrix index
                j = dec_to_ind[cntd_state.lead]
                # total coefficient is phase * sqrt(whatever)
                coeff = phase * _coeff(Nx, Ny, kx, ky, i, j)
                coeff *= _zero_coeff(cntd_state, Nx, Ny, kx, ky)
                try:
                    j_element[j] += coeff
                except KeyError:
                    j_element[j] = coeff
            except exceptions.NotFoundError:  # connecting to a zero state
                pass
    return j_element


def H_ppmm_elements(Nx, Ny, kx, ky, i, l):
    """computes the H++-- elements

    Paramters
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
    i: int
        index of the Bloch state before the Hamiltonian acts on it
    l: int
        range of interaction. 1 for nearest neighbor interation, 2 for
        second neighbors, 3 for third neighbors

    Returns
    --------------------
    j_element: dict
        a dictionary that maps j's to their values for a given i
    """
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i]
    j_element = {}
    bits = _bits(Nx, Ny, l)
    for b1, b2 in zip(*bits):
        upup, downdown = _repeated_spins(state.lead, b1, b2)
        if upup or downdown:
            if upup:
                new_state = state.lead - b1 - b2
                γ = _gamma(Nx, Ny, b1, b2).conjugate()
            elif downdown:
                new_state = state.lead + b1 + b2
                γ = _gamma(Nx, Ny, b1, b2)

            try:
                cntd_state, phase = _find_leading_state(Nx, Ny, kx, ky, new_state)
                j = dec_to_ind[cntd_state.lead]
                coeff = phase * _coeff(Nx, Ny, kx, ky, i, j)
                coeff *= _zero_coeff(cntd_state, Nx, Ny, kx, ky)
                try:
                    j_element[j] += coeff * γ
                except KeyError:
                    j_element[j] = coeff * γ
            except exceptions.NotFoundError:
                pass
    return j_element


def H_pmz_elements(Nx, Ny, kx, ky, i, l):
    """computes the H+-z elements

    Paramters
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
    i: int
        index of the Bloch state before the Hamiltonian acts on it
    l: int
        range of interaction. 1 for nearest neighbor interation, 2 for
        second neighbors, 3 for third neighbors

    Returns
    --------------------
    j_element: dict
        a dictionary that maps j's to their values for a given i
    """
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i]
    j_element = {}
    bits = _bits(Nx, Ny, l)
    for b1, b2 in zip(*bits):
        for _ in range(2):
            if state.lead | b1 == state.lead:
                z_contrib = 0.5
            else:
                z_contrib = -0.5

            if state.lead | b2 == state.lead:
                new_state = state.lead - b2
                γ = _gamma(Nx, Ny, b1, b2).conjugate()
            else:
                new_state = state.lead + b2
                γ = -_gamma(Nx, Ny, b1, b2)

            try:
                cntd_state, phase = _find_leading_state(Nx, Ny, kx, ky, new_state)
                j = dec_to_ind[cntd_state.lead]
                coeff = phase * _coeff(Nx, Ny, kx, ky, i, j)
                coeff *= _zero_coeff(cntd_state, Nx, Ny, kx, ky)
                try:
                    j_element[j] += z_contrib * γ * coeff
                except KeyError:
                    j_element[j] = z_contrib * γ * coeff
            except exceptions.NotFoundError:
                pass

            # switch sites 1 and 2 and repeat
            b1, b2 = b2, b1
    return j_element


@functools.lru_cache(maxsize=None)
def H_z_matrix(Nx, Ny, kx, ky, l):
    """constructs the Hz matrix by calling the H_z_elements function while
    looping over all available i's
    """
    n = len(_bloch_states(Nx, Ny, kx, ky))
    data = np.empty(n)
    for i in range(n):
        data[i] = H_z_elements(Nx, Ny, kx, ky, i, l)
    inds = np.arange(n)
    return sparse.csc_matrix((data, (inds, inds)), shape=(n, n))


def _offdiag_components(Nx, Ny, kx, ky, l, func):
    """constructs the H+-, H++-- and H+-z matrices by calling their
    corresponding functions while looping over all available i's
    """
    n = len(_bloch_states(Nx, Ny, kx, ky))
    row, col, data = [], [], []
    for i in range(n):
        j_elements = func(Nx, Ny, kx, ky, i, l)
        for j, element in j_elements.items():
            row.append(i)
            col.append(j)
            data.append(element)
    return sparse.csc_matrix((data, (row, col)), shape=(n, n))


@functools.lru_cache(maxsize=None)
def H_pm_matrix(Nx, Ny, kx, ky, l):
    return _offdiag_components(Nx, Ny, kx, ky, l, H_pm_elements)


@functools.lru_cache(maxsize=None)
def H_ppmm_matrix(Nx, Ny, kx, ky):
    l = 1
    return _offdiag_components(Nx, Ny, kx, ky, l, H_ppmm_elements)


@functools.lru_cache(maxsize=None)
def H_pmz_matrix(Nx, Ny, kx, ky):
    l = 1
    return 1j * _offdiag_components(Nx, Ny, kx, ky, l, H_pmz_elements)


@functools.lru_cache(maxsize=None)
def _recurring_operators(Nx, Ny, kx, ky):
    H_ppmm = H_ppmm_matrix(Nx, Ny, kx, ky)
    H_pmz = H_pmz_matrix(Nx, Ny, kx, ky)
    return H_ppmm, H_pmz


def hamiltonian_consv_k(Nx, Ny, kx, ky, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0, J2=0, J3=0):
    """construct the full Hamiltonian matrix in the given momentum configuration

    Paramters
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
    H: scipy.sparse.csc_matrix
    """
    H_z1 = H_z_matrix(Nx, Ny, kx, ky, 1)
    H_pm1 = H_pm_matrix(Nx, Ny, kx, ky, 1)
    H_ppmm, H_pmz = _recurring_operators(Nx, Ny, kx, ky)
    nearest_neighbor_terms = J_pm * H_pm1 + J_z * H_z1 + J_ppmm * H_ppmm + J_pmz * H_pmz
    second_neighbor_terms = third_neighbor_terms = 0
    if not J2 == 0:
        H_z2 = H_z_matrix(Nx, Ny, kx, ky, 2)
        H_pm2 = H_pm_matrix(Nx, Ny, kx, ky, 2)
        second_neighbor_terms = J2 * (H_pm2 + J_z / J_pm * H_z2)
    if not J3 == 0:
        H_z3 = H_z_matrix(Nx, Ny, kx, ky, 3)
        H_pm3 = H_pm_matrix(Nx, Ny, kx, ky, 3)
        third_neighbor_terms = J3 * (H_pm3 + J_z / J_pm * H_z3)
    return nearest_neighbor_terms + second_neighbor_terms + third_neighbor_terms
