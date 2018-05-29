#!/usr/bin/python
from scipy.special import psi
from mathematics import *
import numpy as np
import unittest

class TestMathematics(unittest.TestCase):

    def test_digamma_scalar(self):
        x = 2
        self.assertEqual(digamma(x), psi(x))

    def test_digamma_vector(self):
        x = [2,1]
        self.assertTrue(np.array_equal(digamma(x), psi(x)))

    def test_digamma_matrix(self):
        x = np.array([[1,2],[3,4]])
        self.assertTrue(np.array_equal(digamma(x),psi(x)))

if __name__ == '__main__':
    unittest.main()
