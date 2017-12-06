import spinsys as s
import numpy as np
from scipy import sparse
import hamiltonians.quasi_heisenberg.oneleg_blkdiag as ol
from spinsys.quantities import von_neumann_entropy as vne
from spinsys.half import reduced_density_op as rho
from spinsys.quantities import adj_gap_ratio as agr
import time
import datetime


# start = time.time()
### Globals ###
N = 10
c = np.sqrt(2) - 1
edens = [.1, .15, .20, .25, .30, .35, .40, .45, .50]
num_phi = 1000
num_eigs = 200 # {N:E}: {10,30}, {12, 100}, {14, 200}, {16, 200}
print('\n  N = {}'.format(N))
# dhs = [0.15] * 16 + [0.2] * 13 + [0.5] * 2 + [1] * 2  # increments of h
# hs = np.cumsum(dhs)                     # array of h vals
hs = np.linspace(1.2, 2.2, 21)
phis = 2 * np.pi * np.random.random_sample(num_phi) # np.linspace(0, np.pi, num_phi + 1)[:-1]
failed_configs = ['The following configurations failed to yield results:\n']

### body ###
for eden in edens:
    # print("energy density", round(eden, 2))
    # timer = s.utils.timer.Timer(len(phis) * len(hs), mode='moving_average')
    ent_h = np.empty((len(hs), 3))
    agr_h = np.empty((len(hs), 3))
    for i, h in enumerate(hs):
        entropies = []
        adj_gap_ratio = []
        for phi in phis:
            H = ol.H(N, h, c, phi, mode='open')
            try:
                eigs, psis = s.state_generators.generate_eigenpairs(N, H, num_eigs, expand=True, enden=eden)
            except s.exceptions.NoConvergence:
                failed_configs.append('h: {}, phi: {}'
                                      .format(round(h, 1), round(phi, 4)))
                # timer.progress()
                end = time.time()

                continue
            entropies.append([vne(rho(N, N//2, psi)) for psi in psis])
            eigs = np.sort(eigs)    # eigenvalues need to be sorted for AGR
            adj_gap_ratio.append(agr(eigs))
            # end = time.time()
            # print("elapsed: ", str(datetime.timedelta(seconds=(end-start))))
            # timer.progress()
        entropies = np.array(entropies).flatten()
        adj_gap_ratio = np.array(adj_gap_ratio).flatten()
        np.savetxt('./data/entropy_N{}_h{}_e{}.txt'.format(N, round(h, 2), eden),
                   entropies)
        np.savetxt('./data/agr_N{}_h{}_e{}.txt'.format(N, round(h, 2), eden),
                   adj_gap_ratio)
        entropies /= N                      # entropy per site
        ent_h[i, 1] = np.mean(entropies)
        ent_h[i, 2] = np.std(entropies) / np.sqrt(num_phi)
        agr_h[i, 1] = np.mean(adj_gap_ratio)
        agr_h[i, 2] = np.std(adj_gap_ratio) / np.sqrt(num_phi)
    # Save the plots
    ent_h[:, 0] = agr_h[:, 0] = hs
    np.savetxt('./plot_entropy_N{}_{}eden.txt'
               .format(N, eden), ent_h)
    np.savetxt('./plot_agr_N{}_{}eden.txt'.format(N, eden),
               agr_h)
    logfilename = './log_N{}_c{}_{}eigs_{}eden.txt'.format(N, round(c, 2), num_eigs, eden)

    # np.savetxt('./entropy_N{}_c{}_{}eigs_{}eden.txt'
    #            .format(N, round(c, 4), num_eigs, eden), ent_h)
    # np.savetxt('./agr_N{}_c{}_{}eigs_{}eden.txt'.format(N, round(c, 4), num_eigs, eden),
    #            agr_h)
    # logfilename = './log_N{}_c{}_{}eigs_{}eden.txt'.format(N, round(c, 4), num_eigs, eden)
    # with open(logfilename, 'w') as fh:
    #     fh.write('\n'.join(failed_configs))
