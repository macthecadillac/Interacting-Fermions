import spinsys
from spinsys.hamiltonians.aubry_andre_quasi_periodic \
     .one_leg_block_diagonalized import single_blk_hamiltonian
import numpy as np


Ns = [12, 14, 16]
hs = [0.3, 1.0, 1.6, 2.0, 6.0]
num_phi = 500
c = np.sqrt(2)
phis = np.linspace(0, np.pi, num_phi + 1)[:-1]
time_pts = 75
ts = np.logspace(0, np.log10(200), time_pts)
for N in Ns:
    print('\n  Working on N = {}'.format(N))
    datadir = './N{}_timedep/'.format(N)
    logfilename = './log_N{}.txt'.format(N)
    failed_configs = ['The following configurations failed to yield ' +
                      'usable states:']
    timer = spinsys.utils.timer.Timer(num_phi * len(hs))
    for h in hs:
        sgos = []
        for phi in phis:
            H = single_blk_hamiltonian(N, h, c, phi)
            try:
                psi = spinsys.state_generators.generate_product_state(H, tol=1e-3)
            except spinsys.exceptions.StateNotFoundError:
                failed_configs.append('h: {}\tphi: {}'.format(h, phi))
                timer.progress()
                continue
            except spinsys.exceptions.NoConvergence:
                failed_configs.append('h: {}\tphi: {}'.format(h, phi))
                timer.progress()
                continue
            eigvals, eigvecs = np.linalg.eigh(H.toarray())
            psi = spinsys.misc.TimeMachine(eigvals, eigvecs, psi)
            sgos_curr_phi = []
            for t in ts:
                curr_state = psi.evolve_to_time(t)
                sgos_curr_phi.append(spinsys.quantities
                                     .spin_glass_order(N, curr_state))
            sgos.append(sgos_curr_phi)
            timer.progress()
        data = np.array([list(ts)] + sgos).T
        np.savetxt(datadir + 'sgo_h{}_c{}.txt'.format(round(h, 1), round(c, 4)),
                   data)
    log = '\n'.join(failed_configs)
    with open(logfilename, 'w') as fh:
        fh.write(log)
