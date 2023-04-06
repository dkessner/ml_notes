#!/usr/bin/env python
#
# linear_regression
#


# scipy linalg docs
# https://docs.scipy.org/doc/scipy/tutorial/linalg.html


import numpy as np
import scipy
import matplotlib.pyplot as plt


#
# linear model with gaussian noise
#

class LinearFunction:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def __call__(self, x):
        return self.m * x + self.b


def generate_data(x, f, sd_noise=1):
    noise = np.random.normal(0, sd_noise, len(x))
    return f(x) + noise


#
# least squares regression
#

def least_squares(x, y):

    # matrix calculation

    x1t = np.array([x, np.ones_like(x)]) # x1t = (x 1)'
    x1 = np.transpose(x1t)               # x1  = (x 1)
    A = x1t @ x1                         # A is 2x2
    A_inv = np.linalg.inv(A)

    m_est, b_est = A_inv @ x1t @ y

    # scipy least squares for comparison
    solution, _, _, _ = scipy.linalg.lstsq(x1, y)
    print("estimates from scipy:", solution)

    return m_est, b_est


#
# gradient descent
#

def C(x, y, m, b):
    return np.linalg.norm(y - (m*x+b), ord=2)

def dCdm(x, y, m, b):
    return m*x@x + b*x.sum() - x@y

def dCdb(x, y, m, b):
    return (m*x+b - y).sum()

def gradient_descent(x, y):

    m = 0
    b = 0

    #print("initial:", m, b, C(x, y, m, b))

    rate = .0003

    for i in range(10000):
        m -= dCdm(x, y, m, b) * rate
        b -= dCdb(x, y, m, b) * rate

    #print("final:", m, b, C(x, y, m, b))

    return m, b



def main():

    # generate data

    f = LinearFunction(2, 3)
    x = np.linspace(0, 10, 101)
    y = generate_data(x, f)

    # true model line

    x_endpoints = np.array([0, 10])
    y_true = f(x_endpoints)

    # least squares regression 

    m_est, b_est = least_squares(x, y)
    print("least squares estimates (m,b):", str(m_est), str(b_est))

    y_est = m_est * x_endpoints + b_est

    # gradient descent

    m_gd, b_gd = gradient_descent(x, y)
    print("gradient descent estimates (m,b):", str(m_gd), str(b_gd))

    # plot

    plt.plot(x, y, '.')
    plt.plot(x_endpoints, y_true, 'g-')
    plt.plot(x_endpoints, y_est, 'r-')
    plt.ylim([0, 25])
    plt.show()



if __name__ == '__main__':
    main()

