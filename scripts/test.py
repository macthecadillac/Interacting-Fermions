from spinsys.hamiltonians.triangular_lattice_model import hamiltonian
from scipy import sparse
import spinsys


def correlation(vec, Nx, Ny):
    vec = sparse.csc_matrix(vec)
    SiSj = []
    for bond in bonds:
        i, j = bond
        Si = z_mats[i]
        Sj = z_mats[j]
        SiSj.append(Si.dot(Sj))
    return vec.T.conjugate().dot(sum(SiSj)).dot(vec)[0, 0]


Nx = 4
Ny = 3
N = Nx * Ny
σz = spinsys.constructors.sigmaz()
z_mats = [spinsys.half.full_matrix(σz, k, N) for k in range(N)]

bonds = []
vec = spinsys.constructors.SiteVector((0, 0), Nx, Ny)
for i in range(N):
    bonds.append((vec.lattice_index, vec.xhop(1).lattice_index))
    bonds.append((vec.lattice_index, vec.yhop(1).lattice_index))
    bonds.append((vec.lattice_index, vec.xhop(-1).yhop(1).lattice_index))
    vec = vec.next_site()

H = hamiltonian(Nx, Ny)
E, ground_state = sparse.linalg.eigsh(H, k=1, which='SA')
c = correlation(ground_state, Nx, Ny)
print(c)
