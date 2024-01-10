import numpy as np
from hyperspace import train, predict

x1, x2, x3, y = np.loadtxt("pizza_3_vars.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=100000, lr=0.001)

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(30):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))

