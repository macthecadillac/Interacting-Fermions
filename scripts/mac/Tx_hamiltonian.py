import fractions
from itertools import chain
import numpy as np
from spinsys import exceptions, utils
import functools
from hamiltonians.triangular_lattice_model import SiteVector
from collections import namedtuple

Shape = namedtuple('shape', 'x, y')


def bond_list(Nx, Ny):
    N = Nx * Ny
    bonds = []
    vec = SiteVector((0, 0), Nx, Ny)
    for i in range(N):
        bonds.append((vec, vec.xhop(1)))
        bonds.append((vec, vec.yhop(1)))
        bonds.append((vec, vec.xhop(-1).yhop(1)))
        vec = vec.next_site()
    return bonds


def roll_x(state, Nx, Ny):
    """roll to the right"""
    @utils.cache.cache_to_ram
    def xcache(Nx, Ny):
        n = np.arange(0, Nx * Ny, Nx)
        return (2 ** (n + Nx)), 2 ** n, 2 ** Nx, 2 ** (Nx - 1), 2 ** n

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
            states[s].append(bloch_func)
        except KeyError:
            states[s] = [bloch_func]

    N = Nx * Ny
    sieve = np.ones(2 ** N, dtype=np.int8)
    sieve[0] = sieve[-1] = 0
    maxdec = 2 ** N - 1
    states = {Shape(1, 1): [np.array([[0]]), np.array([[maxdec]])]}
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

    if kx == 0 and ky == 0:
        states = list(chain(*zero_k_states.values()))
    elif kx == 0:
        if check_bounds(Ny, ky):
            states = []
            for s in zero_k_states.keys():
                period = Ny // fractions.gcd(Ny, ky)
                if s.y % period == 0:
                    states.extend(zero_k_states[s])
        else:
            raise exceptions.NotFoundError
    elif ky == 0:
        if check_bounds(Nx, kx):
            states = []
            for s in zero_k_states.keys():
                period = Nx // fractions.gcd(Nx, kx)
                if s.x % period == 0:
                    states.extend(zero_k_states[s])
        else:
            raise exceptions.NotFoundError
    elif check_bounds(Nx, kx) and check_bounds(Ny, ky):
        states = []
        for s in zero_k_states.keys():
            periodx = Nx // fractions.gcd(Nx, kx)
            periody = Ny // fractions.gcd(Ny, ky)
            if s.x % periodx == 0 and s.y % periody == 0:
                states.extend(zero_k_states[s])
    else:
        raise exceptions.NotFoundError
    return states


@functools.lru_cache(maxsize=None)
def generate_dec_to_ind_dictionary(Nx, Ny, kx, ky):
    states = bloch_states(Nx, Ny, kx, ky)
    dec_to_ind = {}
    for i, state in enumerate(states):
        state_shape = Shape(x=state.shape[1], y=state.shape[0])
        for num in state:
            dec_to_ind[num] = (i, state_shape)
    return states, dec_to_ind


def all_bloch_states(Nx, Ny):
    states = {}
    max_kx = Nx // 2
    max_ky = Ny // 2
    for kx in range(-max_kx, max_kx + 1):
        for ky in range(-max_ky, max_ky + 1):
            try:
                states[(kx, ky)] = bloch_states(Nx, Ny, kx, ky)
            except exceptions.NotFoundError:
                continue
    return states


@functools.lru_cache(maxsize=None)
def count_same_spins(N, state, l, return_loc=False):
    # edge cases that break down under the kind of binary operation we
    #  use later
    if not (state == 2 ** N - 1 or state == 0):
        # subtraction does a bitwise flip of 1's and 0's. We need this
        #  because subsequent operations are insensitive to patterns of 0's
        bit_inverted_state = 2 ** N - 1 - state
        # the mod operator accounts for periodic boundary conditions
        couplings = 2 ** np.arange(N) + 2 ** (np.arange(l, N + l) % N)
        nup = ndown = 0
        ups = None
        downs = None
        for i in couplings:
            # if a state is unchanged under bitwise OR with the given number,
            #  it has two 1's at those two sites
            if state | i == state:
                nup += 1
                ups = i
            # cannot be merged with the previous statement because bit-flipped
            #  numbers using subtraction are possibly also bit-shifted
            if bit_inverted_state | i == bit_inverted_state:
                ndown += 1
                downs = i
    elif state == 0:
        nup, ndown = 0, N
    else:
        nup, ndown = N, 0  # when state is all 0's

    if return_loc:
        return nup, ndown, ups, downs
    else:
        return nup, ndown


def H_z_elements(N, k, i, l):
    state = generate_dec_to_ind_dictionary(N, k)[0][i][0]
    nup, ndown = count_same_spins(N, state, l)
    same_dir = nup + ndown
    diff_dir = N - same_dir
    return 0.25 * (same_dir - diff_dir)


def H_pm_elements(N, k, i, l):
    states, dec_to_ind = generate_dec_to_ind_dictionary(N, k)
    state = states[i][0]
    i_len = len(states[i])
    bit1 = 2 ** np.arange(N)
    bit2 = 2 ** (np.arange(l, N + 1) % N)
    count = 0  # count of up-downs and down-ups separated by l
    for b1, b2 in zip(bit1, bit2):
        if (state | b1 == state) and (not state | b2 == state):
            count += 1
        if (not state | b1 == state) and (state | b2 == state):
            count += 1
            pm = (b1, b2)
    # bit-flip the state once to see what it is coupled to. We don't need
    #  to do it more than once since such flips will always get to
    #  constituent states of the same bloch state
    connected_state = state + pm[0] - pm[1]
    j, j_len = dec_to_ind[connected_state]
    val = i_len * count / j_len * np.sqrt(j_len / i_len)
    return val, j


def H_ppmm_elements(N, k, i, l):
    state = generate_dec_to_ind_dictionary(N, k)[0][i][0]
    nup, ndown, ups, downs = count_same_spins(N, state, l, return_loc=True)
    if ups is not None:
        connected_state = state - ups
    elif downs is not None:
        connected_state = state + downs
    else:
        raise exceptions.NotFoundError
    j, j_len = dec_to_ind[connected_state]


Nx = 5
Ny = 4

# print(roll_x(8, Nx, Ny))

N = Nx * Ny
states = zero_momentum_states(Nx, Ny)
totlen = 0
for key, val in states.items():
    print('\nlengths = {}\ncount = {}\n'.format(key, len(val)))
    for s in val:
        p = []
        for row in s:
            p.append(list(map(lambda x: format(x, '0{}b'.format(N)), row)))
        print(p)
        # print(s, '\n')
        totlen += np.prod(s.shape)
print('\nSanity check. Total number of product states:', totlen)
print('Total number of zero momentum states: {}\n'
      .format(len(list(chain(*states.values())))))

print('--------------------------------------------------')
states = all_bloch_states(Nx, Ny)
for key, val in states.items():
    print('\nkx, ky = {}\ncount = {}\n'.format(key, len(val)))
    # for s in val:
    #     p = []
    #     for row in s:
    #         p.append(list(map(lambda x: format(x, '0{}b'.format(N)), row)))
    #     print(p)
print('\nTotal number of states:', len(list(chain(*states.values()))))

# N = 5
# k = 0
# i = 0
# state = generate_dec_to_ind_dictionary(N, k)[0][i]
# print(list(map(lambda x: format(x, '0{}b'.format(N)), state)))
# print(H_z_elements(N, k, i, 1))

# i = 2
# val, j = H_pm_elements(N, k, i, 1)
# print(val, j)

# s = int('0110100011100110', 2)
# print(format(s, '016b'))
# print(format(roll_state_to_rightx(s, 8, 2), '016b'))
