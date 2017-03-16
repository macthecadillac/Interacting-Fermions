from spinsys.hamiltonians.quasi_heisenberg import nleg, oneleg
import unittest
from numpy import testing
import numpy as np


class TestNLegH(unittest.TestCase):

    def test_N4_W11_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 4, 1, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1).toarray()

        desired = oneleg.H(N, W1, c1, phi1).toarray()
        testing.assert_array_equal(output, desired)

    def test_N5_W3_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 5, 3, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1).toarray()
        desired = oneleg.H(N, W1, c1, phi1).toarray()
        testing.assert_array_equal(output, desired)

    def test_N8_W6_csqrt2_phi1_1leg_open(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1).toarray()
        desired = oneleg.H(N, W1, c1, phi1).toarray()
        testing.assert_array_equal(output, desired)

    def test_N4_W11_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 4, 1, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()

        desired = oneleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        testing.assert_array_equal(output, desired)

    def test_N5_W3_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 5, 3, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        desired = oneleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        testing.assert_array_equal(output, desired)

    def test_N8_W6_csqrt2_phi1_1leg_periodic(self):
        N, W1, c1, phi1 = 8, 6, np.sqrt(2), 1
        output = nleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        desired = oneleg.H(N, W1, c1, phi1, mode='periodic').toarray()
        testing.assert_array_equal(output, desired)

if __name__ == '__main__':
    unittest.main()
