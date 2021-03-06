import abc
import numpy as np
from scipy.sparse import csc_matrix, eye, kron
from scipy.sparse.linalg import eigsh
from collections import namedtuple

Block = namedtuple('Block', 'side, length, basis_size, block, ops')
key = namedtuple('key', 'side, length')


class Storage():

    blocks = {}

    def __init__(self, init_lblock, init_rblock, init_ops):
        lblock = Block('l', length=1, basis_size=2, block=init_lblock,
                       ops=init_ops)
        rblock = Block('r', length=1, basis_size=2, block=init_rblock,
                       ops=init_ops)

        self.blocks[key('l', 1)] = lblock
        self.blocks[key('r', 1)] = rblock

    @property
    def keys(self):
        return self.blocks.keys()

    def set_item(self, key, block):
        """key is a tuple"""
        self.blocks[key] = block

    def get_item(self, key):
        return self.blocks[key]


class DMRG():

    """The underlying machinary for DMRG. This code is not optimized."""

    def __init__(self, target_m, L, H):
        self.m = target_m
        self.L = L
        self.H = H

    def transform(self, U, A):
        return U.T.conjugate().dot(A).dot(U)

    def enlarge(self, block_key):
        prev_block = self.H.storage.get_item(block_key)
        enl_prev_block = kron(prev_block.block, eye(2))
        block_newsite_interaction = self.H.block_newsite_interaction(block_key)
        new_length = prev_block.length + 1
        new_basis_size = prev_block.basis_size * 2
        new_block = enl_prev_block + block_newsite_interaction
        new_ops = self.H.newsite_ops(new_basis_size)
        return Block(prev_block.side, new_length, new_basis_size, new_block,
                     new_ops)

    def form_super_block(self, enl_sys_block, enl_env_block):
        sys_basis_size = enl_sys_block.basis_size
        env_basis_size = enl_env_block.basis_size
        expanded_sys_block = kron(enl_sys_block.block, eye(env_basis_size))
        expanded_env_block = kron(eye(sys_basis_size), enl_env_block.block)
        site_site_interaction = sum(kron(enl_sys_block.ops[i], enl_env_block.ops[i])
                                    for i in enl_sys_block.ops.keys())
        return expanded_sys_block + expanded_env_block + site_site_interaction

    def update_old_ops(self, sys_block_key, block_key, proj_op):
        sys = True if sys_block_key.side == sys_block_key.side else False
        old_block = self.H.storage.get_item(block_key)
        # pad the old operators if their dimensions are too small
        updated_block_ops = {}
        proj_op_nrow, proj_op_ncol = proj_op.shape
        if sys:
            for i in old_block.ops.keys():
                updated_block_ops[i] = kron(old_block.ops[i], eye(2))
            resized_block = kron(old_block.block, eye(2))
        else:
            for i in old_block.ops.keys():
                updated_block_ops[i] = kron(eye(2), old_block.ops[i])
            resized_block = kron(eye(2), old_block.block)
        # project the old operators onto the new basis
        for i in updated_block_ops.keys():
            updated_block_ops[i] = self.transform(proj_op, updated_block_ops[i])
        projected_block = self.transform(proj_op, resized_block)
        updated_block = Block(
            side=block_key.side,
            length=block_key.length,
            basis_size=proj_op_ncol,
            block=projected_block,
            ops=updated_block_ops
        )
        self.H.storage.set_item(block_key, updated_block)

    def step(self, sys_block_key, env_block_key, grow=False):
        enl_sys_block = self.enlarge(sys_block_key)
        enl_env_block = self.enlarge(env_block_key)
        superblock = self.form_super_block(enl_sys_block, enl_env_block)
        erg, ground_state = eigsh(superblock, k=1, which='SA')
        reshaped_state = ground_state.reshape(enl_sys_block.basis_size, -1)
        rho = reshaped_state.dot(reshaped_state.T.conjugate())
        curr_m = min(self.m, enl_sys_block.basis_size)
        eigs, eigvects = np.linalg.eigh(rho)
        trunc_err = 1 - sum(eigs[-curr_m:])
        proj_op = csc_matrix(eigvects[:, -curr_m:][:, ::-1])
        trunc_sys_block = self.transform(proj_op, enl_sys_block.block)
        trunc_newsite_ops = dict((i, self.transform(proj_op, enl_sys_block.ops[i]))
                                 for i in enl_sys_block.ops.keys())
        for k in self.H.storage.keys:
            if k.side == sys_block_key.side:
                self.update_old_ops(sys_block_key, k, proj_op)
        blocks = [enl_sys_block, enl_env_block] if grow else [enl_sys_block]
        for block in blocks:
            newblock = Block(
                side=block.side,
                length=block.length,
                basis_size=curr_m,
                block=trunc_sys_block,
                ops=trunc_newsite_ops
            )
            newblock_key = key(block.side, block.length)
            self.H.storage.set_item(newblock_key, newblock)
        return erg, trunc_err


class Hamiltonian(abc.ABC):

    """Template for Hamiltonians to work with the DMRG class

    Example implementation:
    >>>from spinsys import sigmax, sigmay, sigmaz
    >>>
    >>>class HeisenbergH(H):
    >>>
    >>>    def __init__(self):
    >>>        self.generators = {'x': sigmax(), 'y': sigmay(), 'z': sigmaz()}
    >>>
    >>>    def initialize_storage(self):
    >>>        init_block = csc_matrix(([], ([], [])), shape=[2, 2])
    >>>        init_ops = self.generators
    >>>        self.storage = Storage(init_block, init_block, init_ops)
    >>>
    >>>    def newsite_ops(self, size):
    >>>        return dict((i, kron(eye(size // 2), self.generators[i]))
    >>>                    for i in self.generators.keys())
    >>>
    >>>    def block_newsite_interaction(self, block_key):
    >>>        block_ops = self.storage.get_item(block_key).ops
    >>>        site_ops = self.generators
    >>>        return sum(kron(block_ops[i], site_ops[i]) for i in site_ops.keys())
    """

    def __init__(self):
        self.initialize_storage()

    @abc.abstractmethod
    def initialize_storage(self):
        pass

    @abc.abstractmethod
    def newsite_ops(self, size):
        pass

    @abc.abstractmethod
    def block_newsite_interaction(self, block_key):
        pass


def finite_system_dmrg(hamiltonian, target_m, L, sweeps=2, debug=False):

    """Performs a finite system DMRG on a given system.

    Parameters:
    --------------------
    hamiltonian: Hamiltonian
        An instantiated class that inherits from the Hamiltonian base class.
    target_m: int
        The number of states to keep during truncation
    L: int
        The final system size
    sweeps: int, optional
        The number of sweeps to perform. Defaults to 2
    debug: bool, optional
        If enabled, eigenenergies of the ground state will be printed
        after every step of the calculation.
    """

    dmrg = DMRG(target_m, L, hamiltonian)
    sys_side = 'l'
    env_side = 'r'
    for sys_len in range(1, L - 2):
        env_len = sys_len if sys_len < L // 2 else L - sys_len - 2
        sys_block_key = key(sys_side, sys_len)
        env_block_key = key(env_side, env_len)
        grow = True if sys_len < L // 2 else False
        erg, trunc_err = dmrg.step(sys_block_key, env_block_key, grow)
        if debug:
            print('sys', sys_len, 'env', env_len, erg[0], trunc_err)
    for sweep in range(sweeps):
        for _ in range(2):  # two half sweeps in one full sweep
            sys_side, env_side = env_side, sys_side
            for sys_len in range(1, L - 2):
                env_len = L - sys_len - 2
                sys_block_key = key(sys_side, sys_len)
                env_block_key = key(env_side, env_len)
                erg, trunc_err = dmrg.step(sys_block_key, env_block_key)
                if debug:
                    print('sys', sys_len, 'env', env_len, erg[0], trunc_err)
