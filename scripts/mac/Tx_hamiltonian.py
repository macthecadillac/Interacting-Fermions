from collections import namedtuple
import fractions
import functools
from itertools import chain, groupby
import numpy as np
from spinsys import exceptions, utils
from hamiltonians.triangular_lattice_model import SiteVector

Shape = namedtuple('shape', 'x, y')


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
def gen_dec_to_ind_dict(Nx, Ny, kx, ky):
    states = bloch_states(Nx, Ny, kx, ky)
    dec_to_ind = {}
    for i, state in enumerate(states):
        state_shape = Shape(x=state.shape[1], y=state.shape[0])
        for num in state.flatten():
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


def count_same_spins(N, state, l):
    # subtraction does a bitwise flip of 1's and 0's. We need this
    #  because subsequent operations are insensitive to patterns of 0's
    inverted_state = 2 ** N - 1 - state
    # the mod operator accounts for periodic boundary conditions
    couplings = 2 ** np.arange(N) + 2 ** (np.arange(l, N + l) % N)
    nup = ndown = 0
    ups = downs = None
    for i in couplings:
        # if a state is unchanged under bitwise OR with the given number,
        #  it has two 1's at those two sites
        if (state | i == state).any():
            # state is an int when counting in the y-direction while a numpy
            #  array when counting in the x-direction. This is admittedly an
            #  ugly hack to make this function, in a sense, "polymorphic".
            bool_arr = np.array([state | i == state]).flatten()
            for w, b in enumerate(bool_arr):
                nup += b
                if b:
                    ups = 2 ** (w * N)
        # cannot be merged with the previous statement because bit-flipped
        #  numbers using subtraction are possibly also bit-shifted
        if (inverted_state | i == inverted_state).any():
            bool_arr = np.array([inverted_state | i == inverted_state]).flatten()
            for w, b in enumerate(bool_arr):
                ndown += b
                if b:
                    downs = 2 ** (w * N)
    return nup, ndown, ups, downs


def H_z_elements(Nx, Ny, kx, ky, i, l):
    N = Nx * Ny
    # i is the index of a certain state in our ordering (which is better
    # characterized as unordered while deterministic--it always stays the same).
    # The last [0] part picks out one particular number from the state since any
    # constituent product state within a Bloch state contains enough information
    # for us to locate/recreate the entire Bloch state
    state = gen_dec_to_ind_dict(Nx, Ny, kx, ky)[0][i][0, 0]
    # x-direction
    sliced_state = slice_state(state, Nx, Ny)
    nup_x, ndown_x, ups_x, downs_x = count_same_spins(Nx, sliced_state, l)
    nup_y, ndown_y, ups_y, downs_y = count_same_spins(N, state, l * Nx)
    same_dir = nup_x + nup_y + ndown_x + ndown_y
    diff_dir = 2 * N - same_dir
    return 0.25 * (same_dir - diff_dir)


# def count_spin_flips(state, bit1, bit2):
#     count = 0
#     for b1, b2 in zip(bit1, bit2):
#         # we don't really need to count up-downs since for every up-down there
#         # must be a down-up and vice versa
#         bool_arr1 = np.invert(np.array([state | b1 == state]).flatten())
#         bool_arr2 = np.array([state | b2 == state]).flatten()
#         for i, (bool1, bool2) in enumerate(zip(bool_arr1, bool_arr2)):
#             if bool1 and bool2:
#                 count += 2
#                 pm = np.array([b1, b2])
#                 pos = i
#     return count, pm, pos


def count_spin_flips(pdstate, bit1, bit2):
    count = 0
    downup_loc = []
    updown_loc = []    # list of locations of spin-flips
    for b1, b2 in zip(bit1, bit2):
        if (pdstate | b1 == pdstate) and (not pdstate | b2 == pdstate):
            count += 1
            updown_loc.append((b1, b2))
        if (not pdstate | b1 == pdstate) and (pdstate | b2 == pdstate):
            downup_loc.append((b1, b2))
    return count, updown_loc, downup_loc


def H_pm_elements(Nx, Ny, kx, ky, i, l):
    states, dec_to_ind = gen_dec_to_ind_dict(Nx, Ny, kx, ky)
    full_state = states[i]
    state = full_state[0, 0]  # only one product state is needed. See above.
    i_shape = Shape(x=full_state.shape[1], y=full_state.shape[0])
    # x-direction
    bit1 = 2 ** (np.arange(l, Nx + 1) % Nx)
    bit2 = 2 ** np.arange(Nx)
    sliced_state = slice_state(state, Nx, Ny)
    xcount = 0
    updown_locs = []
    downup_locs = []
    for s, pdstate in enumerate(sliced_state):
        cnt, udl, dul = count_spin_flips(pdstate, bit1, bit2)
        xcount += cnt
        if udl:
            udl = np.array(udl)
            updown_locs.extend(udl * 2 ** (s * Nx))
        if dul:
            dul = np.array(dul)
            downup_locs.extend(dul * 2 ** (s * Nx))

    # y-direction
    N = Nx * Ny
    bit1 = 2 ** np.arange(N)
    bit2 = 2 ** (np.arange(l * Nx, N + 1) % N)
    ycount, updown_loc, downup_loc = count_spin_flips(state, bit1, bit2)
    updown_locs.extend(np.array(updown_loc))
    downup_locs.extend(np.array(downup_loc))

    # bit-flip the state once to see what it is coupled to. We don't need to do
    # it to more than one product state since such flips will always get to
    # some constituent product state of the same bloch state
    # FIXME: connect to multiple states
    connected_states = []
    for updown_loc in updown_locs:
        connected_states.append(state - updown_loc[0] + updown_loc[1])
    for downup_loc in downup_locs:
        connected_states.append(state + downup_loc[0] - downup_loc[1])

    contd_states_cnt = []
    for connected_state in connected_states:
        j, j_shape = dec_to_ind[connected_state]
        contd_states_cnt.append((j, j_shape))
    contd_states_cnt.sort()
    count_connect = groupby(contd_states_cnt)
    for item, i in count_connect:
        print(item, len(tuple(i)))
    vals = []
    val = i_shape.x * count / j_shape.x * np.sqrt(j_shape.x / i_shape.x)
    return val, j


# # FIXME: algorithm needs to be completely reevaulated
# def H_pm_elements_y(Nx, Ny, kx, ky, i, l):
#     # pretty much identical to the x-direction version
#     states, dec_to_ind = gen_dec_to_ind_dict(Nx, Ny, kx, ky)
#     full_state = states[i]
#     state = full_state[0, 0]
#     i_shape = Shape(x=full_state.shape[1], y=full_state.shape[0])
#     N = Nx * Ny
#     bit1 = 2 ** np.arange(N)
#     bit2 = 2 ** (np.arange(l * Nx, N + 1) % N)
#     count, pm, pos = count_spin_flips(state, bit1, bit2)

#     connected_state = state + pm[0] - pm[1]
#     j, j_shape = dec_to_ind[connected_state]
#     val = i_shape.y * count / j_shape.y * np.sqrt(j_shape.y / i_shape.y)
#     return val, j


def H_ppmm_elements(N, k, i, l):
    state = gen_dec_to_ind_dict(N, k)[0][i][0]
    nup, ndown, ups, downs = count_same_spins(N, state, l)
    if ups is not None:
        connected_state = state - ups
    elif downs is not None:
        connected_state = state + downs
    else:
        raise exceptions.NotFoundError
    j, j_len = dec_to_ind[connected_state]


Nx = 3
Ny = 4

# print(roll_x(8, Nx, Ny))

# N = Nx * Ny
# states = zero_momentum_states(Nx, Ny)
# totlen = 0
# for key, val in states.items():
#     print('\nlengths = {}\ncount = {}\n'.format(key, len(val)))
#     for s in val:
#         p = []
#         for row in s:
#             p.append(list(map(lambda x: format(x, '0{}b'.format(N)), row)))
#         print(p)
#         # print(s, '\n')
#         totlen += np.prod(s.shape)
# print('\nSanity check. Total number of product states:', totlen)
# print('Total number of zero momentum states: {}\n'
#       .format(len(list(chain(*states.values())))))

# print('--------------------------------------------------')
# states = all_bloch_states(Nx, Ny)
# for key, val in states.items():
#     print('\nkx, ky = {}\ncount = {}\n'.format(key, len(val)))
#     # for s in val:
#     #     p = []
#     #     for row in s:
#     #         p.append(list(map(lambda x: format(x, '0{}b'.format(N)), row)))
#     #     print(p)
# print('\nTotal number of Bloch states:', len(list(chain(*states.values()))))

# N = Nx * Ny
# kx = ky = 0
# i = 6
# l = 2
# state = gen_dec_to_ind_dict(Nx, Ny, kx, ky)[0][i]
# p = []
# print('State:')
# for row in state:
#     p.append(list(map(lambda x: format(x, '0{}b'.format(N)), row)))
# print(p)
# print('Element: {}, l: {}'.format(H_z_elements(Nx, Ny, kx, ky, i, l), l))

i = 17
kx = ky = 0
l = 1
state = gen_dec_to_ind_dict(Nx, Ny, kx, ky)[0][i]
p = []
print('Nx = {}, Ny = {}'.format(Nx, Ny))
print('State:')
for row in state:
    p.append(list(map(lambda x: format(x, '0{}b'.format(Nx * Ny)), row)))
print(p)
val, j = H_pm_elements(Nx, Ny, kx, ky, i, l)
print(val, j)
