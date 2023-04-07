#!/usr/bin/env python
#
# nn.py
#


import numpy as np
import scipy
import matplotlib.pyplot as plt


#
# VectorFunction(m,n): R^m -> R^n
#   - derivative() returns nxm matrix
#

class VectorFunction:
    def __init__(self, len_in, len_out):
        self.len_in = len_in
        self.len_out = len_out
    def __call__(self, x):
        assert len(x) == self.len_in
        return np.zeros(self.len_out)
    def derivative(self, x):
        return None
    def derivative_parameter(self, x):
        return None


def test_vector_function():
    print("test_vector_function()")
    f = VectorFunction(3, 4)
    x = np.array([1, 3, 5])
    assert (f(x) == np.zeros(4)).all()

#
# LinearTransformation(A): R^m -> R^n
#   - defined by the nxm matrix A
#   - derivative() returns A
#

class LinearTransformation(VectorFunction):
    def __init__(self, A):
        self.A = A
        len_out, len_in = A.shape
        super().__init__(len_in, len_out)
    def __call__(self, x):
        return self.A @ x
    def derivative(self, x):
        return self.A.copy()
    def derivative_parameter(self, x):
        return x
    def add(self, dA):
        assert self.A.shape == dA.shape
        self.A += dA
    def __repr__(self):
        return str(self.A)


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


#
# Network(T)
#   - represents the composite VectorFunction
#       T[0] -> T[1] -> ... -> T[n-1]
#

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
            self.dT_dparam = []
            self.final = None
        
    def __call__(self, x):

        result = self.CallResult()

        for t in self.T:
            # apply each transformation in the list,
            y = t(x)

            # save intermediate results and derivatives
            result.x.append(y)
            result.dT.append(t.derivative(x))
            result.dT_dparam.append(t.derivative_parameter(x))

            x = y            

        # convenience reference to final values
        result.final = result.x[-1] 

        return result

    class CostGradient:
        def __init__(self):
            self.C = None
            self.dC_dparam = []

    def calculate_cost_and_gradient(self, x, call_result, y):

        result = self.CostGradient()
        result.C = np.linalg.norm(y - call_result.final, ord=2)

        # TODO: backprop
        result.dC_dparam = [np.transpose(x @ np.transpose(call_result.final - y))]

        return result


    def apply_gradient_step(self, dC_dparam, rate):
        for t, dCdt in zip(self.T, dC_dparam):
            if dCdt is not None:
                t.add(-dCdt*rate)
        

    def gradient_descent(self, x, y):

        rate = .0003
        max_iterations = 5000
        C_current = None
        dC_threshold = -1e-5

        for i in range(max_iterations):

            # calculate function and gradient

            call_result = self(x)
            cost_gradient = self.calculate_cost_and_gradient(x, call_result, y)

            # exit condition: |dC| small

            dC = cost_gradient.C - C_current \
                    if C_current is not None \
                    else dC_threshold

            C_current = cost_gradient.C

            if dC > 0:
                print("[Network:gradient_descent()] dC:", dC)
                print("[Network:gradient_descent()] Error: dC > 0")
                return

            if dC > dC_threshold:
                print("[Network:gradient_descent()] break: dC > threshold")
                print("[Network:gradient_descent()] iterations:", i)
                break

            #print("C:", cost_gradient.C)
            #print("T:", self.T)
            #print("gradient:", cost_gradient.dC_dparam)

            # move in direction of gradient

            self.apply_gradient_step(cost_gradient.dC_dparam, rate)



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

    result = n(e)
    print("n(e):", result.__dict__)
    print("n(e).final:", n(e).final)
    assert (n(e).x[-1] == [[2,4], [6,8], [10,12]]).all()




class SimpleLinearNetwork(Network):
    def __init__(self, m=0.0, b=0.0):
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
    assert (y.final == 2*x+3).all()


def test_network_cost():
    
    print("\ntest_network_cost()")

    f = lambda x : 2*x + 3
    x = np.linspace(0, 10, 101)
    noise = np.random.normal(0, 1, len(x))
    y = f(x) + noise

    l = SimpleLinearNetwork()

    xt = np.array([np.ones_like(x), x]) # xt = (x)' (2 row vectors)
    call_result = l(xt)
    cost_gradient = l.calculate_cost_and_gradient(xt, call_result, y)

    print("C:", cost_gradient.C)

    b, m = l.T[0].A[0]
    print("b, m:", b, m)

    C_expected = np.linalg.norm(y - (m*x+b), ord=2)
    print("C expected:", C_expected)
    
    assert abs(cost_gradient.C - C_expected) < 1e-6


def test_gradient_descent():

    print("\ntest_gradient_descent()")

    f = lambda x : 2*x + 3
    x = np.linspace(0, 10, 101)
    noise = np.random.normal(0, 1, len(x))
    y = f(x) + noise

    l = SimpleLinearNetwork()

    xt = np.array([np.ones_like(x), x]) # xt = (x)' (2 row vectors)
 
    l.gradient_descent(xt, y)

    b_est, m_est = l.T[0].A[0]
    print("b_est, m_est:", b_est, m_est)

    # plot

    x_endpoints = np.array([0, 10])
    y_true = f(x_endpoints)

    y_est = m_est * x_endpoints + b_est

    plt.plot(x, y, '.')
    plt.plot(x_endpoints, y_true, 'g-')
    plt.plot(x_endpoints, y_est, 'r-')
    plt.ylim([0, 25])
    plt.show()



def run_tests():
    test_vector_function()
    test_linear_transformation()
    test_network()
    test_simple_linear_network()
    test_network_cost()

def main():
    run_tests()
    test_gradient_descent()


if __name__ == '__main__':
    main()

