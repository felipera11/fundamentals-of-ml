import numpy as np

# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# performing Forward-Propagation
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)

# Calling the predict() function
def classify(X, w):
    y_hat = forward(X, w)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

# Computing Loss 
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.sum(first_term + second_term) / X.shape[0]

# Calculating gradient
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

# Adding a bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

# print function for printing each iteration's log
def report(iteration, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)
    n_test_examples = Y_test.shape[0]
    matches = matches * 100.0 / n_test_examples
    training_loss = loss(X_train, Y_train, w)
    if (iteration == 1) or (iteration % 1500 == 0) or (iteration == 9999): 
        print("%d - Loss: %.20f, %.2f%%" % (iteration, training_loss, matches))

# training phase
def train(X_train, Y_train, X_test, Y_test, iterations, lr):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))
    for iteration in range(iterations):
        report(iteration, X_train, Y_train, X_test, Y_test, w)
        w -= gradient(X_train, Y_train, w) * lr
    report(iteration, X_train, Y_train, X_test, Y_test, w)
    return w