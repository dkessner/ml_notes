#!/usr/bin/env python
#
# logistic regression
#

import numpy as np
import scipy
import matplotlib.pyplot as plt

from numpy import exp, log


#
# reference:
#   https://en.wikipedia.org/wiki/Logistic_regression
#


x = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# general logistic function

def sigma(t, b0, b1):
    return 1/(1 + exp(-b0+(-b1)*t))

# log-likelihood and gradient

def L(x, y, b0, b1):
    s = sigma(x, b0, b1)
    return (y*log(s) + (1-y)*log(1-s)).sum()

def dLdb0(x, y, b0, b1):
    s = sigma(x, b0, b1)
    return (y-s).sum()

def dLdb1(x, y, b0, b1):
    s = sigma(x, b0, b1)
    return (x*(y-s)).sum()



def main():

    print(x, y)

    b0 = 0
    b1 = 1
    rate = .05
    epsilon = 1e-8

    for i in range(5000):
        print(b0, b1, L(x,y,b0,b1))
        db0 = dLdb0(x, y, b0, b1) * rate
        db1 = dLdb1(x, y, b0, b1) * rate
        b0 += db0
        b1 += db1
        if abs(db0)<epsilon and abs(db1)<epsilon:
            print("iterations:", i+1)
            break

    plt.plot(x, y, '.')
    plt.plot(x, sigma(x, b0, b1), '-')
    plt.show()




if __name__ == '__main__':
    main()


