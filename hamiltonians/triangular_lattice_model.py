from collections import namedtuple
import copy
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


def _find_state(Nx, Ny, kx, ky, state):
    def _rollx(j):
        for ix in range(Nx):
            if j in dec_to_ind.keys():
                return j, ix
            else:
                j = roll_x(j, Nx, Ny)

    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    j = state
    for iy in range(Ny):
        try:
            j, ix = _rollx(j)
            full_state = ind_to_dec[dec_to_ind[j]]
            φx = np.exp(1j * 2 * np.pi * kx * ix / full_state.shape.x)
            φy = np.exp(1j * 2 * np.pi * ky * iy / full_state.shape.y)
            phase = φx * φy
            return full_state, phase
        except TypeError:
            j = roll_y(j, Nx, Ny)


@functools.lru_cache(maxsize=None)
def _gamma(Nx, Ny, b1, b2):
    m = int(round(np.log2(b1)))
    n = int(round(np.log2(b2)))
    vec1 = SiteVector.from_index(m, Nx, Ny)
    vec2 = SiteVector.from_index(n, Nx, Ny)
    ang = vec1.angle_with(vec2)
    return np.exp(1j * ang)


@functools.lru_cache(maxsize=None)
def _bits(Nx, Ny, l):
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
    if abs(kx) > Nx // 2 or abs(ky) > Ny // 2:
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
                if ky * s.y % Ny == 0:
                    for state in zero_k_states[s]:
                        states.append(state)
                        shapes.append(s)
        else:
            raise exceptions.NotFoundError
    elif ky == 0:
        if check_bounds(Nx, kx):
            for s in zero_k_states.keys():
                if kx * s.x % Nx == 0:
                    for state in zero_k_states[s]:
                        states.append(state)
                        shapes.append(s)
        else:
            raise exceptions.NotFoundError
    elif check_bounds(Nx, kx) and check_bounds(Ny, ky):
        for s in zero_k_states.keys():
            if (kx * s.x % Nx == 0) and (ky * s.y % Ny == 0):
                for state in zero_k_states[s]:
                    states.append(state)
                    shapes.append(s)
    else:
        raise exceptions.NotFoundError
    table = BlochTable(states, shapes)
    table.sort()
    return table


@functools.lru_cache(maxsize=None)
def _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky):
    states = bloch_states(Nx, Ny, kx, ky)
    dec = states.dec
    nstates = len(dec)
    inds = list(range(nstates))
    dec_to_ind = dict(zip(dec, inds))
    ind_to_dec = dict(zip(inds, states))
    return ind_to_dec, dec_to_ind


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


def H_z_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    bit1, bit2 = _bits(Nx, Ny, l)
    same_dir = 0
    for b1, b2 in zip(bit1, bit2):
        upup, downdown = _repeated_spins(state, b1, b2)
        same_dir += upup + downdown
    diff_dir = len(bit1) - same_dir
    return 0.25 * (same_dir - diff_dir)


@functools.lru_cache(maxsize=None)
def _coeff(Nx, Ny, kx, ky, i, j):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    orig_state = ind_to_dec[i]
    cntd_state = ind_to_dec[j]
    orig_state_len = orig_state.shape.x * orig_state.shape.y
    cntd_state_len = cntd_state.shape.x * cntd_state.shape.y
    coeff = np.sqrt(orig_state_len / cntd_state_len)
    if not kx == 0:
        if cntd_state.shape.x * kx % orig_state.shape.x != 0 or orig_state.shape.x * kx % cntd_state.shape.x != 0:
            coeff = 0
    if not ky == 0:
        if cntd_state.shape.y * ky % orig_state.shape.y != 0 or orig_state.shape.y * ky % cntd_state.shape.y != 0:
            coeff = 0
    return coeff


def _format(N, state):
    return format(state, '0{}b'.format(N))


# def _find_T_invariant_set(Nx, Ny, i):
#     b = [i]
#     j = i
#     while True:
#         j = roll_x(j, Nx, Ny)
#         if not j == i:
#             b.append(j)
#         else:
#             break
#     b = np.array(b)
#     bloch_func = [b]
#     u = b.copy()
#     while True:
#         u = roll_y(u, Nx, Ny)
#         if not np.sort(u)[0] == np.sort(b)[0]:
#             bloch_func.append(u)
#         else:
#             break
#     bloch_func = np.array(bloch_func, dtype=np.int64)
#     shape = bloch_func.shape
#     return bloch_func.flatten().repeat((Nx * Ny) // (shape[0] * shape[1]))


def H_pm_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i]
    j_element = {}
    bits = _bits(Nx, Ny, l)
    # new_states = []
    for b1, b2 in zip(*bits):
        updown, downup = _exchange_spin_flips(state.dec, b1, b2)
        if updown or downup:
            if updown:
                new_state = state.dec - b1 + b2
            elif downup:
                new_state = state.dec + b1 - b2
            # print(_find_T_invariant_set(Nx, Ny, new_state))
            # new_states.extend(_find_T_invariant_set(Nx, Ny, new_state))
            # new_states.append(new_state)
    # print(new_states)

    # for new_state in new_states:
            try:
                # find what connected state it is if the state we got from bit-
                #  flipping is not in our records
                cntd_state, phase = _find_state(Nx, Ny, kx, ky, new_state)
                # phase1 = phase
                j = dec_to_ind[cntd_state.dec]
                coeff = phase * _coeff(Nx, Ny, kx, ky, i, j)

                # N = Nx * Ny
                # if (i, j) == (18, 6) or (i, j) == (6, 18):
                #     print('orig state: {}\tnew state: {}\tconnected state: {}'
                #           .format(_format(N, state), _format(N, new_state),
                #                   _format(N, cntd_state.dec)))
                #     print('connected state shape: {}'.format(cntd_state.shape))
                #     print((i, j), cntd_state.shape.x, Nx, cntd_state.shape.y, Ny,
                #           phase1, phase1 * _coeff(Nx, Ny, kx, ky, i, j), '\n')

                # print(i, j, phase, coeff, cntd_state.shape.x, Nx, cntd_state.shape.y, Ny)
                # N = Nx * Ny
                # if i in [6, 35]:
                #     print('{}   \torig state: {}\tnew state: {}\tconnected state: {}'
                #           .format((i, j), _format(N, state), _format(N, new_state),
                #                   _format(N, cntd_state), '\n'))
                # if (i, j) == (35, 6):
                #     print(coeff, phase)
                #     print(_format(N, state), _format(N, new_state),
                #           _format(N, cntd_state), _coeff(Nx, Ny, kx, ky, i, j), '\n')
                # elif (i, j) == (6, 35):
                #     print('\t\t\t\t', coeff)
                #     N = Nx * Ny
                #     print(_format(N, state), _format(N, new_state),
                #           _format(N, cntd_state), _coeff(Nx, Ny, kx, ky, i, j), '\n')
                try:
                    j_element[j] += coeff
                except KeyError:
                    j_element[j] = coeff
            except TypeError:  # connecting to a zero state
                pass
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
                γ = _gamma(Nx, Ny, b1, b2).conjugate()
            elif downdown:
                new_state = state + b1 + b2
                γ = _gamma(Nx, Ny, b1, b2)

            try:
                connected_state, phase = _find_state(Nx, Ny, kx, ky, new_state)
                j = dec_to_ind[connected_state.dec]
                coeff = phase * _coeff(Nx, Ny, kx, ky, i, j)
                try:
                    j_element[j] += coeff * γ
                except KeyError:
                    j_element[j] = coeff * γ
            except TypeError:
                pass
    return j_element


def H_pmz_elements(Nx, Ny, kx, ky, i, l):
    ind_to_dec, dec_to_ind = _gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    j_element = {}
    bits = _bits(Nx, Ny, l)
    for b1, b2 in zip(*bits):
        for _ in range(2):
            if state | b1 == state:
                z_contrib = 0.5
            else:
                z_contrib = -0.5

            if state | b2 == state:
                new_state = state - b2
                γ = _gamma(Nx, Ny, b1, b2).conjugate()
            else:
                new_state = state + b2
                γ = -_gamma(Nx, Ny, b1, b2)

            try:
                connected_state, phase = _find_state(Nx, Ny, kx, ky, new_state)
                j = dec_to_ind[connected_state.dec]
                coeff = phase * _coeff(Nx, Ny, kx, ky, i, j)
                try:
                    j_element[j] += z_contrib * γ * coeff
                except KeyError:
                    j_element[j] = z_contrib * γ * coeff
            except TypeError:
                pass

            b1, b2 = b2, b1
    return j_element


@functools.lru_cache(maxsize=None)
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
        for j, element in j_elements.items():
            row.append(i)
            col.append(j)
            data.append(element)
    return sparse.csc_matrix((data, (row, col)), shape=(n, n))


# I'm partially applying the functions manually because the syntax of
#  python does not lend itself well to the use of closures
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
