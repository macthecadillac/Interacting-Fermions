from collections import namedtuple
import copy
import fractions
import functools
import numpy as np
from scipy import sparse
from spinsys import constructors, half, dmrg, exceptions


Shape = namedtuple('Shape', 'x, y')
Momentum = namedtuple('Momentum', 'kx, ky')
# dec is the decimanl representation of one constituent product state of a Bloch function
BlochFunc = namedtuple('BlochFunc', 'dec, shape')


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
    bonds_by_dist = [[], [], []]
    for i in range(N):
        nearest_neighbor = vec.nearest_neighboring_sites
        second_neighbor = vec.second_neighboring_sites
        third_neighbor = vec.third_neighboring_sites
        neighbors = [nearest_neighbor, second_neighbor, third_neighbor]
        for leap, bonds in enumerate(bonds_by_dist):
            for n in neighbors[leap]:
                bond = (vec, n)
                bonds.append(bond)
        vec = vec.next_site()
    return bonds_by_dist


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
    second_neighbor_terms = J2 * (H_pm2 + J_z / J_pm * H_z2)
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

    def __init__(self, states, shapes):
        self.data = [states, shapes]
        self.dataT = list(zip(*self.data))

    def __getitem__(self, i):
        return BlochFunc(*self.dataT[i])

    def __len__(self):
        return len(self.data[0])

    def sort(self):
        table = zip(*self.data)
        table = sorted(table, key=lambda x: x[0])
        self.data = list(zip(*table))
        self.dataT = list(zip(*self.data))

    @property
    def dec(self):
        return self.data[0]


def slice_state(state, Nx, Ny):
    # slice a given state into different "numbers" with each corresponding to
    #  one leg in the x-direction.
    n = np.arange(0, Nx * Ny, Nx)
    return state % (2 ** (n + Nx)) // (2 ** n)


@functools.lru_cache(maxsize=None)
def _roll_x_aux(Nx, Ny):
    n = np.arange(0, Nx * Ny, Nx)
    a = 2 ** (n + Nx)
    b = 2 ** n
    c = 2 ** Nx
    d = 2 ** (Nx - 1)
    e = 2 ** n
    return a, b, c, d, e


def roll_x(state, Nx, Ny):
    """roll to the right"""
    # slice state into Ny equal slices
    a, b, c, d, e = _roll_x_aux(Nx, Ny)
    s = state % a // b
    s = (s * 2) % c + s // d
    return (e).dot(s)


@functools.lru_cache(maxsize=None)
def _roll_y_aux(Nx, Ny):
    return 2 ** Nx, 2 ** (Nx * (Ny - 1))


def roll_y(state, Nx, Ny):
    """roll up"""
    xdim, pred_totdim = _roll_y_aux(Nx, Ny)
    tail = state % xdim
    return state // xdim + tail * pred_totdim


def _exchange_spin_flips(state, b1, b2):
    updown = downup = False
    if (state | b1 == state) and (not state | b2 == state):
        updown = True
    if (not state | b1 == state) and (state | b2 == state):
        downup = True
    return updown, downup


def _repeated_spins(state, b1, b2):
    upup = downdown = False
    if (state | b1 == state) and (state | b2 == state):
        upup = True
    if (not state | b1 == state) and (not state | b2 == state):
        downdown = True
    return upup, downdown


def _find_state(Nx, Ny, state, dec_to_ind):
    def _rollx(j):
        for _ in range(Nx):
            j = roll_x(j, Nx, Ny)
            if j in dec_to_ind.keys():
                return j

    jstate = state
    for _ in range(Ny):
        jstate = roll_y(jstate, Nx, Ny)
        j = _rollx(jstate)
        if j is not None:
            return j


@functools.lru_cache(maxsize=None)
def _gamma_from_bits(Nx, Ny, b1, b2):
    N = Nx * Ny
    m = int(round(np.log2(b1)))
    n = int(round(np.log2(b2)))
    diff = abs(m - n)
    det = min(N - diff, diff)
    if det == Nx:
        γ = np.exp(-1j * 2 * np.pi / 3)
    elif det == 1:
        γ = 1
    else:
        γ = np.exp(1j * 2 * np.pi / 3)
    return γ


@functools.lru_cache(maxsize=None)
def _bits(Nx, Ny, l):
    N = Nx * Ny
    # interaction along x-direction
    xbit1 = 2 ** (np.arange(l, N + l) % Nx + np.arange(0, N, Nx).repeat(Nx))
    xbit2 = 2 ** (np.arange(N) % Nx + np.arange(0, N, Nx).repeat(Nx))
    # interaction along y-direction
    ybit1 = 2 ** np.arange(N)
    ybit2 = 2 ** (np.arange(l * Nx, N + l * Nx) % N)
    # interaction along the diagonal direction
    dbit1 = np.roll(np.arange(N) % Nx, -1) + np.arange(0, N, Nx).repeat(Nx)
    dbit1 = np.array([roll_y(b, Nx, Ny) for b in 2 ** dbit1])
    dbit2 = 2 ** (np.arange(N) % Nx + np.arange(0, N, Nx).repeat(Nx))
    bit1 = np.concatenate((xbit1, ybit1, dbit1))
    bit2 = np.concatenate((xbit2, ybit2, dbit2))
    return bit1, bit2


@functools.lru_cache(maxsize=None)
def _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky):
    states = bloch_states(Nx, Ny, kx, ky)
    dec = states.dec
    nstates = len(dec)
    inds = list(range(nstates))
    dec_to_ind = dict(zip(dec, inds))
    ind_to_dec = dict(zip(inds, states))
    return ind_to_dec, dec_to_ind


@functools.lru_cache(maxsize=None)
def zero_momentum_states(Nx, Ny):
    """only returns the representative configuration. All other constituent
    product states within the same Bloch state could be found by repeatedly
    applying the translation operator (the roll_something functions)
    """
    def find_T_invariant_set(i):
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
            shape = bloch_func.shape
            s = Shape(x=shape[1], y=shape[0])
            states[s].append(i)
        except KeyError:
            states[s] = [i]

    N = Nx * Ny
    sieve = np.ones(2 ** N, dtype=np.int8)
    sieve[0] = sieve[-1] = 0
    maxdec = 2 ** N - 1
    states = {Shape(1, 1): [0, maxdec]}
    for i in range(maxdec + 1):
        if sieve[i]:
            find_T_invariant_set(i)
    return states


@functools.lru_cache(maxsize=None)
def bloch_states(Nx, Ny, kx, ky):
    def check_bounds(N, k):
        return k > 0 or (k < 0 and ((not -k == N // 2) or (not N % 2 == 0)))

    zero_k_states = zero_momentum_states(Nx, Ny)
    if kx > Nx // 2 or ky > Ny // 2:
        raise exceptions.NotFoundError

    states, shapes = [], []
    if kx == 0 and ky == 0:
        for s in zero_k_states.keys():
            for state in zero_k_states[s]:
                states.append(state)
                shapes.append(s)
    elif kx == 0:
        if check_bounds(Ny, ky):
            for s in zero_k_states.keys():
                period = Ny // fractions.gcd(Ny, ky)
                if s.y % period == 0:
                    for state in zero_k_states[s]:
                        states.append(state)
                        shapes.append(s)
        else:
            raise exceptions.NotFoundError
    elif ky == 0:
        if check_bounds(Nx, kx):
            for s in zero_k_states.keys():
                period = Nx // fractions.gcd(Nx, kx)
                if s.x % period == 0:
                    for state in zero_k_states[s]:
                        states.append(state)
                        shapes.append(s)
        else:
            raise exceptions.NotFoundError
    elif check_bounds(Nx, kx) and check_bounds(Ny, ky):
        for s in zero_k_states.keys():
            periodx = Nx // fractions.gcd(Nx, kx)
            periody = Ny // fractions.gcd(Ny, ky)
            if s.x % periodx == 0 and s.y % periody == 0:
                for state in zero_k_states[s]:
                    states.append(state)
                    shapes.append(s)
    else:
        raise exceptions.NotFoundError
    table = BlochTable(states, shapes)
    table.sort()
    return table


def all_bloch_states(Nx, Ny):
    states = {}
    max_kx = Nx // 2
    max_ky = Ny // 2
    for kx in range(-max_kx, max_kx + 1):
        for ky in range(-max_ky, max_ky + 1):
            try:
                states[Momentum(kx, ky)] = bloch_states(Nx, Ny, kx, ky)
            except exceptions.NotFoundError:
                continue
    return states


def count_same_spins(N, state, l):
    # subtraction does a bitwise flip of 1's and 0's. We need this
    #  because subsequent operations are insensitive to patterns of 0's
    inverted_state = 2 ** N - 1 - state
    # the mod operator accounts for periodic boundary conditions
    couplings = 2 ** np.arange(N) + 2 ** (np.arange(l, N + l) % N)
    nup = ndown = 0
    for i in couplings:
        # if a state is unchanged under bitwise OR with the given number,
        #  it has two 1's at those two sites
        if (state | i == state):
            nup += 1
        # cannot be merged with the previous statement because bit-flipped
        #  numbers using subtraction are possibly also bit-shifted
        if (inverted_state | i == inverted_state):
            ndown += 1
        # TODO: fix the over-counting checking for arbitrary l
        # don't over-count interactions for chain with len=2
        if len(couplings) == 2:
            break
    return nup, ndown


def H_z_elements(Nx, Ny, kx, ky, i, l):
    # "l" is the leap. l = 1 for nearest neighbor coupling and so on.
    N = Nx * Ny
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    # x-direction
    sliced_state = slice_state(state, Nx, Ny)
    nup_x = ndown_x = 0
    for leg in sliced_state:
        u, d = count_same_spins(Nx, leg, l)
        nup_x += u
        ndown_x += d
    # y-direction
    nup_y, ndown_y = count_same_spins(N, state, l * Nx)
    same_dir = nup_x + nup_y + ndown_x + ndown_y
    # TODO: fix the over-counting checking for arbitrary l
    nx = Nx if not Nx == 2 else 1  # prevent over-counting interactions
    ny = Ny if not Ny == 2 else 1
    diff_dir_x = nx * Ny - nup_x - ndown_x
    diff_dir_y = Nx * ny - nup_y - ndown_y
    diff_dir = diff_dir_x + diff_dir_y
    return 0.25 * (same_dir - diff_dir)


def H_pm_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    j_element = {}
    bits = _bits(Nx, Ny, l)
    for b1, b2 in zip(*bits):
        updown, downup = _exchange_spin_flips(state, b1, b2)
        if updown or downup:
            if updown:
                new_state = state - b1 + b2
            elif downup:
                new_state = state + b1 - b2

            N = Nx * Ny
            print(updown, downup)
            if new_state not in dec_to_ind.keys():
                connected_state = _find_state(Nx, Ny, new_state, dec_to_ind)
            else:
                connected_state = new_state

            print('\nbits:', format(b1, '0{}b'.format(N)),
                  format(b2, '0{}b'.format(N)))
            print(format(state, '0{}b'.format(N)),
                  format(new_state, '0{}b'.format(N)),
                  connected_state)
            print(format(connected_state, '0{}b'.format(N)))
            j = dec_to_ind[connected_state]
            try:
                j_element[j] += 1
            except KeyError:
                j_element[j] = 1
    return j_element


def H_ppmm_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    j_element = {}
    bits = _bits(Nx, Ny, l)
    for b1, b2 in zip(*bits):
        upup, downdown = _repeated_spins(state, b1, b2)
        if upup or downdown:
            if upup:
                new_state = state - b1 - b2
                γ = _gamma_from_bits(Nx, Ny, b1, b2).conjugate()
            elif downdown:
                new_state = state + b1 + b2
                γ = _gamma_from_bits(Nx, Ny, b1, b2)

            # find what connected state it is if the state we got from bit-
            #  flipping is not in our records
            if new_state not in dec_to_ind.keys():
                connected_state = _find_state(Nx, Ny, new_state, dec_to_ind)
            else:
                connected_state = new_state

            j = dec_to_ind[connected_state]
            try:
                j_element[j] += γ
            except KeyError:
                j_element[j] = γ
    return j_element


def H_pmz_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    j_element = {}
    bits = _bits(Nx, Ny, l)
    for b1, b2 in zip(*bits):
        for _ in range(2):
            if state | b1 == state:
                sgn = 1
            else:
                sgn = -1

            if state | b2 == state:
                new_state = state - b2
                γ = _gamma_from_bits(Nx, Ny, b1, b2).conjugate()
            else:
                new_state = state + b2
                γ = -_gamma_from_bits(Nx, Ny, b1, b2)

            if new_state not in dec_to_ind.keys():
                connected_state = _find_state(Nx, Ny, new_state, dec_to_ind)
            else:
                connected_state = new_state

            j = dec_to_ind[connected_state]
            try:
                j_element[j] += sgn * γ
            except KeyError:
                j_element[j] = sgn * γ

            b1, b2 = b2, b1
    return j_element


def H_z_matrix(Nx, Ny, kx, ky, l):
    n = len(bloch_states(Nx, Ny, kx, ky))
    data = np.empty(n)
    for i in range(n):
        data[i] = H_z_elements(Nx, Ny, kx, ky, i, l)
    inds = np.arange(n)
    return sparse.csc_matrix((data, (inds, inds)), shape=(n, n))


def _offdiag_components(Nx, Ny, kx, ky, l, func):
    n = len(bloch_states(Nx, Ny, kx, ky))
    row, col, data = [], [], []
    for i in range(n):
        j_elements = func(Nx, Ny, kx, ky, i, l)
        for j, count in j_elements.items():
            row.append(i)
            col.append(j)
            data.append(count)
    return sparse.csc_matrix((data, (row, col)), shape=(n, n))


# I'm partially applying the functions manually because the syntax of
#  python does not lend itself well to the use of closures
def H_pm_matrix(Nx, Ny, kx, ky, l):
    return _offdiag_components(Nx, Ny, kx, ky, l, H_pm_elements)


def H_ppmm_matrix(Nx, Ny, kx, ky):
    l = 1
    return _offdiag_components(Nx, Ny, kx, ky, l, H_ppmm_elements)


def H_pmz_matrix(Nx, Ny, kx, ky):
    l = 1
    return 1j * _offdiag_components(Nx, Ny, kx, ky, l, H_pmz_elements)


@functools.lru_cache(maxsize=None)
def hamiltonian_consv_k_components(Nx, Ny, kx, ky):
    H_z1 = H_z_matrix(Nx, Ny, kx, ky, 1)
    H_z2 = H_z_matrix(Nx, Ny, kx, ky, 2)
    H_z3 = H_z_matrix(Nx, Ny, kx, ky, 3)
    print('####################      l = 1      ####################')
    H_pm1 = H_pm_matrix(Nx, Ny, kx, ky, 1)
    print('####################      l = 2      ####################')
    H_pm2 = H_pm_matrix(Nx, Ny, kx, ky, 2)
    print('####################      l = 3      ####################')
    H_pm3 = H_pm_matrix(Nx, Ny, kx, ky, 3)
    H_ppmm = H_ppmm_matrix(Nx, Ny, kx, ky)
    H_pmz = H_pmz_matrix(Nx, Ny, kx, ky)
    return H_z1, H_z2, H_z3, H_pm1, H_pm2, H_pm3, H_ppmm, H_pmz


def hamiltonian_consv_k(Nx, Ny, kx, ky, J_pm=0, J_z=0, J_ppmm=0, J_pmz=0, J2=0, J3=0):
    H_components = hamiltonian_consv_k_components(Nx, Ny, kx, ky)
    H_z1, H_z2, H_z3, H_pm1, H_pm2, H_pm3, H_ppmm, H_pmz = H_components
    nearest_neighbor_terms = J_pm * H_pm1 + J_z * H_z1 + J_ppmm * H_ppmm + J_pmz * H_pmz
    second_neighbor_terms = J2 * (H_pm2 + J_z / J_pm * H_z2)
    third_neighbor_terms = J3 * (H_pm3 + J_z / J_pm * H_z3)
    return nearest_neighbor_terms + second_neighbor_terms + third_neighbor_terms
