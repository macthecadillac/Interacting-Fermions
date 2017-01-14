from spinsys.hamiltonians import aubry_andre_quasi_periodic as a
from spinsys.half.block import similarity_trans_matrix
import unittest
from numpy import testing
import numpy as np


class TestNLegBlockDiagonalized(unittest.TestCase):

    def test_N4_h1_csqrt2_phi1_1leg(self):
        N, h, c, phi = 4, 1, np.sqrt(2), 1
        output = a.nleg_block_diagonalized.single_block(N, h, c, phi,
                                                        I=1).toarray()
        desired = a.block_diagonalized.blk_full(N, h, c, 0, phi).toarray()
        testing.assert_array_equal(output, desired)

    def test_N5_h3_csqrt2_phi1_1leg(self):
        N, h, c, phi = 5, 3, np.sqrt(2), 1
        output = a.nleg_block_diagonalized.single_block(N, h, c, phi,
                                                        I=1).toarray()
        desired = a.block_diagonalized.blk_full(N, h, c, 0, phi).toarray()
        testing.assert_array_equal(output, desired)

    def test_N8_h6_csqrt2_phi1_1leg(self):
        N, h, c, phi = 8, 6, np.sqrt(2), 1
        output = a.nleg_block_diagonalized.single_block(N, h, c, phi,
                                                        I=1).toarray()
        desired = a.block_diagonalized.blk_full(N, h, c, 0, phi).toarray()
        testing.assert_array_equal(output, desired)


if __name__ == '__main__':
    unittest.main()
