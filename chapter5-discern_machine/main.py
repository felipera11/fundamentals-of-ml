import numpy as np
from logistic_regression import train, test

# Prepare data
x1, x2, x3, y = np.loadtxt("police.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=1000000000, lr=0.00001)

# Test it
test(X, Y, w)