import numpy as np
import spinsys as s
from spinsys.hamiltonians.aubry_andre_quasi_periodic \
    .one_leg_block_diagonalized import single_blk_hamiltonian as hamiltonian


# Parameters
c = np.sqrt(2) - 1
N = 18
num_phi = 1
num_eigs = 200
print('\n  N = {}'.format(N))
dhs = [0.15] * 16 + [0.2] * 13 + [0.5] * 2 + [1] * 2  # increments of h
hs = np.cumsum(dhs)[2:3]                     # array of h vals
phis = np.linspace(0, np.pi, num_phi + 1)[:-1]3
failed_configs = ['The following configurations failed to yield results:\n']

# Body
timer = s.utils.timer.Timer(len(phis) * len(hs), mode='moving_average')
ent_h = np.empty((len(hs), 2))
agr_h = np.empty((len(hs), 2))
for i, h in enumerate(hs):
    entropies = []
    adj_gap_ratio = []
    for phi in phis:
        H = hamiltonian(N, h, c, phi, mode='open')
        try:
            eigs, psis = s.state_generators.generate_eigenpairs(N, H, num_eigs)
        except s.exceptions.NoConvergence:
            failed_configs.append('h: {}, phi: {}'
                                  .format(round(h, 1), round(phi, 4)))
            timer.progress()
            continue
        entropies.append([s.quantities.von_neumann_entropy(N, psi)
                          for psi in psis])
        eigs = np.sort(eigs)    # eigenvalues need to be sorted for AGR
        adj_gap_ratio.append(s.quantities.adj_gap_ratio(eigs))
        timer.progress()
    entropies = np.array(entropies).flatten()
    adj_gap_ratio = np.array(adj_gap_ratio).flatten()
    np.savetxt('./N{}/entropy_h{}_c{}.txt'.format(N, round(h, 1), round(c, 4)),
               entropies)
    np.savetxt('./N{}/agr_h{}_c{}.txt'.format(N, round(h, 1), round(c, 4)),
               adj_gap_ratio)
    entropies /= N                      # entropy per site
    ent_h[i, 1] = np.mean(entropies)
    agr_h[i, 1] = np.mean(adj_gap_ratio)

# Save the plots
ent_h[:, 0] = agr_h[:, 0] = hs
np.savetxt('./entropy_N{}_c{}_{}eigs.txt'
           .format(N, round(c, 4), num_eigs), ent_h)
np.savetxt('./agr_N{}_c{}_{}eigs.txt'.format(N, round(c, 4), num_eigs),
           agr_h)
logfilename = './log_N{}_c{}_{}eigs.txt'.format(N, round(c, 4), num_eigs)
with open(logfilename, 'w') as fh:
    fh.write('\n'.join(failed_configs))
