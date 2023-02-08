#!/usr/bin/env python3
#
# hello_numpy
#
# https://numpy.org/doc/stable/user/quickstart.html
#

import numpy as np

a = np.arange(15).reshape(3, 5)

print("a:\n", a)

print("a.shape:", a.shape)
print("a.ndim:", a.ndim)
print("a.dtype.name:", a.dtype.name)
print("a.itemsize:", a.itemsize)
print("a.size:", a.size)
print("type(a):", type(a))

print()
b = np.array([1.0, 3.0, 5.0])
print("b:", b)
print("2*b:", 2*b)
print("elementwise product b*b:", b*b)
print("matrix product b@b:", b@b)
print("type(b):", type(b))
print("b.dtype:", b.dtype)

print()
print("np.zeros(3,4):\n", np.zeros((3,4)))
print("np.ones(3,2):\n", np.ones((3,2), dtype=np.int16))
print("np.arange(10, 30, 5):", np.arange(10, 30, 5))
print("np.linspace(0, 2, 9):", np.linspace(0, 2, 9))

