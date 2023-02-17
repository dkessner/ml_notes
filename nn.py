#!/usr/bin/env python
#
# nn.py
#


import numpy as np
import scipy
import matplotlib.pyplot as plt


class VectorFunction:
    def __init__(self, len_in, len_out):
        self.len_in = len_in
        self.len_out = len_out
    def __call__(self, x):
        assert len(x) == self.len_in
        return np.zeros(self.len_out)


class LinearTransformation(VectorFunction):
    def __init__(self, A):
        self.A = A
        len_out, len_in = A.shape
        super().__init__(len_in, len_out)
    def __call__(self, x):
        return self.A @ x



def main():

    f = VectorFunction(3, 4)
    x = np.array([1, 3, 5])
    print("x:", x)
    print("f(x):", f(x))

    A = np.array([[1, 3, 5], [2, 4, 6]])
    print("A:", A)
    T = LinearTransformation(A)
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    print("T(e1):", T(e1))
    print("T(e2):", T(e2))
    print("T(e3):", T(e3))




if __name__ == '__main__':
    main()

