import unittest
from spinsys import constructors
import numpy as np
from numpy import testing


class TestRaising(unittest.TestCase):
    """Test spinsys.constructors.raising()"""
    def test_spin_one_half(self):
        output = constructors.raising(0.5).toarray()
        desired = np.array([[0, 1], [0, 0]])
        testing.assert_array_equal(output, desired)

    def test_spin_one(self):
        output = constructors.raising(1).toarray()
        desired = np.sqrt(2) * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        testing.assert_array_equal(output, desired)

    def test_spin_three_halves(self):
        output = constructors.raising(1.5).toarray()
        desired = np.sqrt([[0, 3, 0, 0],
                           [0, 0, 4, 0],
                           [0, 0, 0, 3],
                           [0, 0, 0, 0]])
        testing.assert_array_equal(output, desired)


class TestLowering(unittest.TestCase):
    """Test spinsys.constructors.lowering()"""
    def test_spin_one_half(self):
        output = constructors.lowering(0.5).toarray()
        desired = np.array([[0, 0], [1, 0]])
        testing.assert_array_equal(output, desired)

    def test_spin_one(self):
        output = constructors.lowering(1).toarray()
        desired = np.sqrt(2) * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        testing.assert_array_equal(output, desired)

    def test_spin_three_halves(self):
        output = constructors.lowering(1.5).toarray()
        desired = np.sqrt([[0, 0, 0, 0],
                           [3, 0, 0, 0],
                           [0, 4, 0, 0],
                           [0, 0, 3, 0]])
        testing.assert_array_equal(output, desired)


class TestSigmaz(unittest.TestCase):
    """Test spinsys.constructors.sigmaz()"""
    def test_spin_one_half(self):
        output = constructors.sigmaz(0.5).toarray()
        desired = 0.5 * np.array([[1, 0], [0, -1]])
        testing.assert_array_equal(output, desired)

    def test_spin_one(self):
        output = constructors.sigmaz(1).toarray()
        desired = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
        testing.assert_array_equal(output, desired)


if __name__ == '__main__':
    unittest.main()
