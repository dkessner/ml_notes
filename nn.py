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


class Network(VectorFunction):
    def __init__(self, T):
        self.T = T # list of transformations (VectorFunction objects)

        # sanity check
        for i, t in enumerate(T):
            assert isinstance(t, VectorFunction)
            if i>0:
                assert T[i-1].len_out == t.len_in

        super().__init__(T[0].len_in, T[-1].len_out)
        
    def __call__(self, x):
        # apply each transformation in the list
        result = x
        for t in self.T:
            result = t(result)
        return result


def test_vector_function():
    print("test_vector_function()")
    f = VectorFunction(3, 4)
    x = np.array([1, 3, 5])
    assert (f(x) == np.zeros(4)).all()


def test_linear_transformation():
    print("test_linear_transformation()")
    A = np.array([[1, 3, 5], [2, 4, 6]])
    print("A:\n", A)
    T = LinearTransformation(A)
    e = np.eye(3)
    print("T(e[0]):", T(e[0]))
    print("T(e[1]):", T(e[1]))
    print("T(e[2]):", T(e[2]))
    assert (T(e[0]) == [1,2]).all()
    assert (T(e[1]) == [3,4]).all()
    assert (T(e[2]) == [5,6]).all()


def test_network():
    print("test_network()")

    A = np.array([[1,2],[3,4],[5,6]])
    B = np.eye(3) * 2

    n = Network([LinearTransformation(A), 
                 LinearTransformation(B)])

    assert n.len_in == 2
    assert n.len_out == 3

    e = np.eye(2)
    print(n(e[0]))
    print(n(e[1]))
    assert (n(e[0]) == [2,6,10]).all()
    assert (n(e[1]) == [4,8,12]).all()
    print(n(e))
    assert (n(e) == [[2,4], [6,8], [10,12]]).all()



def run_tests():
    test_vector_function()
    test_linear_transformation()
    test_network()


def main():
    run_tests()


if __name__ == '__main__':
    main()

