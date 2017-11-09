from scipy.sparse import csc_matrix, kron, eye
from spinsys import dmrg, constructors
from hamiltonians.triangular_lattice_model import DMRG_Hamiltonian


class HeisenbergH():

    def __init__(self):
        self.generators = {
            'x': constructors.sigmax(),
            'y': constructors.sigmay(),
            'z': constructors.sigmaz()
        }
        init_block = csc_matrix(([], ([], [])), shape=[2, 2])
        init_ops = self.generators
        self.storage = dmrg.Storage(init_block, init_block, init_ops)

    def newsite_ops(self, size):
        return dict((i, kron(eye(size // 2), self.generators[i]))
                    for i in self.generators.keys())

    def block_newsite_interaction(self, block_key):
        block_ops = self.storage.get_item(block_key).ops
        site_ops = self.generators
        return sum(kron(block_ops[i], site_ops[i]) for i in site_ops.keys())


# Nx = 4
# Ny = 4
# N = Nx * Ny
# m = 50
# J_z = 1
# J_pm = 0.5
# H = DMRG_Hamiltonian(Nx, Ny, J_z=J_z, J_pm=J_pm)
# dmrg.finite_system_dmrg(H, m, N, debug=True)
L = 20
target_m = 50
sweeps = 2
H = HeisenbergH()
dmrg.finite_system_dmrg(H, target_m, L, debug=True)
