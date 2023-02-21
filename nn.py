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
    def derivative(self, x):
        return None


def test_vector_function():
    print("test_vector_function()")
    f = VectorFunction(3, 4)
    x = np.array([1, 3, 5])
    assert (f(x) == np.zeros(4)).all()


class LinearTransformation(VectorFunction):
    def __init__(self, A):
        self.A = A
        print("A.shape:", A.shape)
        len_out, len_in = A.shape
        super().__init__(len_in, len_out)
    def __call__(self, x):
        return self.A @ x
    def derivative(self, x):
        return self.A.copy()


def test_linear_transformation():
    print("\ntest_linear_transformation()")
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


class Network(VectorFunction):
    def __init__(self, T):
        self.T = T # list of transformations (VectorFunction objects)

        # sanity check
        for i, t in enumerate(T):
            assert isinstance(t, VectorFunction)
            if i>0:
                assert T[i-1].len_out == t.len_in

        super().__init__(T[0].len_in, T[-1].len_out)

    class CallResult:
        def __init__(self):
            self.x = []
            self.dT = []
            self.dxdw = []
            self.final = None
        
    def __call__(self, x):

        # apply each transformation in the list,
        # save intermediate results and derivatives

        result = self.CallResult()

        for t in self.T:
            y = t(x)
            result.x.append(y)
            result.dT.append(t.derivative(x))
            result.dxdw.append(x) # TODO: t.dtdw(x)
            x = y            

        result.final = result.x[-1] # convenience reference to final values
        return result

    # TODO:
    #def calculate_cost_and_gradient(x, y):

    # TODO:
    # def apply_gradient_step()

    # TODO:
    # def gradient_descent(x, y):
    #     loop:
    #           calculate_cost_and_gradient(x, y)
    #           apply_gradient_step
    #



def test_network():
    print("\ntest_network()")

    A = np.array([[1,2],[3,4],[5,6]])
    B = np.eye(3) * 2

    n = Network([LinearTransformation(A), 
                 LinearTransformation(B)])

    assert n.len_in == 2
    assert n.len_out == 3

    e = np.eye(2)

    print("n(e0).final:", n(e[0]).final)
    print("n(e1).final:", n(e[1]).final)
    assert (n(e[0]).final == [2,6,10]).all()
    assert (n(e[1]).final == [4,8,12]).all()

    print("n(e).final:", n(e).final)
    assert (n(e).x[-1] == [[2,4], [6,8], [10,12]]).all()


class SimpleLinearNetwork(Network):
    def __init__(self, m=0, b=0):
        T = LinearTransformation(np.array([[b, m]]))
        super().__init__([T])


def test_simple_linear_network():
    print("\ntest_simple_linear_network()")
    l = SimpleLinearNetwork(2, 3)
    x = np.array(range(11))
    xt = np.array([np.ones_like(x), x]) # xt = (x)' (2 row vectors)
    y = l(xt)
    print("xt:", xt)
    print("y.final:", y.final)


def run_tests():
    test_vector_function()
    test_linear_transformation()
    test_network()
    test_simple_linear_network()


def main():
    run_tests()


if __name__ == '__main__':
    main()

