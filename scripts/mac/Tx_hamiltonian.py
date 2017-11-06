import fractions
from itertools import chain
import numpy as np
from spinsys.exceptions import NotFoundError
import functools
from hamiltonians.triangular_lattice_model import SiteVector


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


def zero_momentum_states(N):
    def genfunc(i):
        bloch_func = [i]
        j = i
        while True:
            j = (j * 2) % maxdec
            if not j == i:
                bloch_func.append(j)
                sieve[j] = False
            else:
                break
        bloch_func.sort()
        try:
            l = len(bloch_func)
            states[l].append(bloch_func)
        except KeyError:
            states[l] = [bloch_func]

    sieve = np.ones(2 ** N)
    sieve[0] = sieve[-1] = 0
    maxdec = 2 ** N - 1
    states = {1: [[0], [maxdec]]}
    for i in range(maxdec + 1):
        if sieve[i]:
            genfunc(i)
    return states


def bloch_states(N, k):
    zero_k_states = zero_momentum_states(N)
    if k > N // 2:
        raise NotFoundError

    if k == 0:
        states = list(chain(*zero_k_states.values()))
    elif k > 0 or (k < 0 and ((not -k == N // 2) or (not N % 2 == 0))):
        states = []
        for l in zero_k_states.keys():
            period = N // fractions.gcd(N, k)
            if l % period == 0:
                states.extend(zero_k_states[l])
    else:
        raise NotFoundError
    return states


@functools.lru_cache(maxsize=None)
def generate_dec_to_ind_dictionary(N, k):
    states = sorted(bloch_states(N, k))
    dec_to_ind = {}
    for i, state in enumerate(states):
        state_len = len(state)
        for num in state:
            dec_to_ind[num] = (i, state_len)
    return states, dec_to_ind


def all_bloch_states(N):
    states = {}
    max_k = N // 2
    for k in range(-max_k, max_k + 1):
        try:
            states[k] = bloch_states(N, k)
        except NotFoundError:
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
        raise NotFoundError
    j, j_len = dec_to_ind[connected_state]


# N = 6
# states = zero_momentum_states(N)
# totlen = 0
# for key, val in states.items():
#     print('\nlength = {}\ncount = {}\n'.format(key, len(val)))
#     for s in val:
#         print(s)
#         totlen += len(s)
# print('\nSanity check. Total number of product states:', totlen)
# print('Total number of zero momentum states:{}\n'
#       .format(len(list(chain(*states.values())))))
# print('--------------------------------------------------')
# states = all_bloch_states(N)
# for key, val in states.items():
#     print('\nk = {}\ncount = {}\n'.format(key, len(val)))
#     for s in val:
#         print(s)
# print('\nTotal number of states:', len(list(chain(*states.values()))))

# N = 5
# k = 0
# i = 0
# state = generate_dec_to_ind_dictionary(N, k)[0][i]
# print(list(map(lambda x: format(x, '0{}b'.format(N)), state)))
# print(H_z_elements(N, k, i, 1))

# i = 2
# val, j = H_pm_elements(N, k, i, 1)
# print(val, j)
