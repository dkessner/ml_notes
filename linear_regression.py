#!/usr/bin/env python
#
# linear_regression
#


# scipy linalg docs
# https://docs.scipy.org/doc/scipy/tutorial/linalg.html


import numpy as np
import scipy
import matplotlib.pyplot as plt


class LinearFunction:
    def __init__(self, m, b):
        self.m = m
        self.b = b
    def __call__(self, x):
        return self.m * x + self.b


def generate_data(x, f, sd_noise=1):
    noise = np.random.normal(0, sd_noise, len(x))
    return f(x) + noise


# data

f = LinearFunction(2, 3)
x = np.linspace(0, 10, 101)
y = generate_data(x, f)

# true model line

x_endpoints = np.array([0, 10])
y_endpoints = f(x_endpoints)

# regression (manual matrix calculation)

x1t = np.array([x, np.ones_like(x)]) # x1t = (x 1)'
x1 = np.transpose(x1t)               # x1  = (x 1)
A = x1t @ x1                         # A is 2x2
A_inv = np.linalg.inv(A)

m_est, b_est = A_inv @ x1t @ y
print("regression estimates m: " + str(m_est) + "b: " + str(b_est))

y_endpoints_est = m_est * x_endpoints + b_est

# regression (scipy least squares)

solution, _, _, _ = scipy.linalg.lstsq(x1, y)
print("estimates from scipy:", solution)

# plot

plt.plot(x, y, '.')
plt.plot(x_endpoints, y_endpoints, 'g-')
plt.plot(x_endpoints, y_endpoints_est, 'r-')
plt.ylim([0, 25])
plt.show()


