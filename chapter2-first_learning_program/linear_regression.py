import numpy as np

def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)


def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        if i % 300 == 0:
            print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr, b) < current_loss: # Updating weight
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss: # Updating weight
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss: # Updating bias
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss: # Updating bias
            b -= lr
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)