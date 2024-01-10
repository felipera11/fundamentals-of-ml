import numpy as np
# predict funtion
def predict(X, w):
    return np.matmul(X, w)

# loss function
def loss(X, Y, w):
   return np.average((predict(X, w) - Y) ** 2)

# gradient function
def gradient(X, Y, w):
    return 2 * np.matmul(X.T, predict(X, w) - Y) / X.shape[0]

def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w