from collections import namedtuple
import fractions
import functools
from itertools import chain, groupby
import numpy as np
from spinsys import exceptions, utils
from hamiltonians.triangular_lattice_model import SiteVector

Shape = namedtuple('Shape', 'x, y')
Momentum = namedtuple('Momentum', 'kx, ky')
# dec is the decimanl representation of one constituent product state of a Bloch function
BlochFunc = namedtuple('BlochFunc', 'dec, shape')


class Bond:

    def __init__(self, site1, site2):
        self.site1 = site1
        self.site2 = site2
        self.indices = (site1, site2)

    def __len__(self):
        return abs(self.site1.lattice_index - self.site2.lattice_index)


def bond_list(Nx, Ny):
    N = Nx * Ny
    bonds = []
    vec = SiteVector((0, 0), Nx, Ny)
    for i in range(N):
        bonds.append(Bond(vec, vec.xhop(1)))
        bonds.append(Bond(vec, vec.yhop(1)))
        bonds.append(Bond(vec, vec.xhop(-1).yhop(1)))
        vec = vec.next_site()
    return bonds


def slice_state(state, Nx, Ny):
    # slice a given state into different "numbers" with each corresponding to
    #  one leg in the x-direction.
    n = np.arange(0, Nx * Ny, Nx)
    return state % (2 ** (n + Nx)) // (2 ** n)


def roll_x(state, Nx, Ny):
    """roll to the right"""
    @utils.cache.cache_to_ram
    def xcache(Nx, Ny):
        n = np.arange(0, Nx * Ny, Nx)
        a = 2 ** (n + Nx)
        b = 2 ** n
        c = 2 ** Nx
        d = 2 ** (Nx - 1)
        e = 2 ** n
        return a, b, c, d, e

    # slice state into Ny equal slices
    a, b, c, d, e = xcache(Nx, Ny)
    s = state % a // b
    s = (s * 2) % c + s // d
    return (e).dot(s)


def roll_y(state, Nx, Ny):
    """roll up"""
    @utils.cache.cache_to_ram
    def ycache(Nx, Ny):
        return 2 ** Nx, 2 ** (Nx * (Ny - 1))

    xdim, pred_totdim = ycache(Nx, Ny)
    tail = state % xdim
    return state // xdim + tail * pred_totdim


@functools.lru_cache(maxsize=None)
def zero_momentum_states(Nx, Ny):
    """only returns the representative configuration. All other constituent
    product states within the same Bloch state could be found by repeatedly
    applying the translation operator (the roll_something functions)
    """
    def find_T_invariant_set(i):
        j = i
        xlen = ylen = 1
        while True:
            j = roll_x(j, Nx, Ny)
            if not j == i:
                sieve[j] = 0
                xlen += 1
            else:
                break
        while True:
            j = roll_y(j, Nx, Ny)
            if not j == i:
                sieve[j] = 0
                ylen += 1
            else:
                break
        try:
            s = Shape(x=xlen, y=ylen)
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


def bloch_states(Nx, Ny, kx, ky):
    def check_bounds(N, k):
        return k > 0 or (k < 0 and ((not -k == N // 2) or (not N % 2 == 0)))

    zero_k_states = zero_momentum_states(Nx, Ny)
    if kx > Nx // 2 or ky > Ny // 2:
        raise exceptions.NotFoundError

    states = []
    if kx == 0 and ky == 0:
        for s in zero_k_states.keys():
            for state in zero_k_states[s]:
                states.append(BlochFunc(dec=state, shape=s))
    elif kx == 0:
        if check_bounds(Ny, ky):
            for s in zero_k_states.keys():
                period = Ny // fractions.gcd(Ny, ky)
                if s.y % period == 0:
                    for state in zero_k_states[s]:
                        states.append(BlochFunc(dec=state, shape=s))
        else:
            raise exceptions.NotFoundError
    elif ky == 0:
        if check_bounds(Nx, kx):
            for s in zero_k_states.keys():
                period = Nx // fractions.gcd(Nx, kx)
                if s.x % period == 0:
                    for state in zero_k_states[s]:
                        states.append(BlochFunc(dec=state, shape=s))
        else:
            raise exceptions.NotFoundError
    elif check_bounds(Nx, kx) and check_bounds(Ny, ky):
        for s in zero_k_states.keys():
            periodx = Nx // fractions.gcd(Nx, kx)
            periody = Ny // fractions.gcd(Ny, ky)
            if s.x % periodx == 0 and s.y % periody == 0:
                for state in zero_k_states[s]:
                    states.append(BlochFunc(dec=state, shape=s))
    else:
        raise exceptions.NotFoundError
    return sorted(states, key=lambda x: x.dec)


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


@functools.lru_cache(maxsize=None)
def gen_ind_dec_conv_dicts(Nx, Ny, kx, ky):
    states = bloch_states(Nx, Ny, kx, ky)
    dec = (ψ.dec for ψ in states)
    nstates = len(states)
    inds = list(range(nstates))
    dec_to_ind = dict(zip(dec, inds))
    ind_to_dec = dict(zip(inds, states))
    return ind_to_dec, dec_to_ind


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
    ind_to_dec, dec_to_ind = gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
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


def spin_flips(state, b1, b2):
    updown = downup = False
    if (state | b1 == state) and (not state | b2 == state):
        updown = True
    if (not state | b1 == state) and (state | b2 == state):
        downup = True
    return updown, downup


def find_state(Nx, Ny, state, dec_to_ind):
    def _rollx(j):
        for _ in range(Nx):
            j = roll_x(j, Nx, Ny)
            if j in dec_to_ind.keys():
                return jstate

    jstate = state
    for _ in range(Ny):
        jstate = roll_y(jstate, Nx, Ny)
        j = _rollx(jstate)
        if j is not None:
            return j


def H_pm_elements(Nx, Ny, kx, ky, i, l):
    N = Nx * Ny
    ind_to_dec, dec_to_ind = gen_ind_dec_conv_dicts(Nx, Ny, kx, ky)
    state = ind_to_dec[i].dec
    print(format(state, '0{}b'.format(N)), '\n')
    connected_states = []
    # spin flips in x-direction
    xbit1 = 2 ** (np.arange(l, Nx + 1) % Nx)
    xbit2 = 2 ** np.arange(Nx)
    # spin flips in y-direction
    ybit1 = 2 ** np.arange(N)
    ybit2 = 2 ** (np.arange(l * Nx, N + 1) % N)
    xbits = zip(xbit1, xbit2)
    ybits = zip(ybit1, ybit2)
    # add in γij
    for bits in [xbits, ybits]:
        for b1, b2 in bits:
            updown, downup = spin_flips(state, b1, b2)
            if updown:
                new_state = state - b1 + b2
            if downup:
                new_state = state + b1 - b2

            if new_state not in dec_to_ind.keys():
                connected_state = find_state(Nx, Ny, state, dec_to_ind)
            else:
                connected_state = new_state
            connected_states.append(connected_state)
    connected_states.sort()
    connected_states_cnt = groupby(connected_states)
    return list(map(lambda x: (x[0], len(list(x[1]))), connected_states_cnt))
