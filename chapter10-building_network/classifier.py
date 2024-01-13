import numpy as np

# Applying Logistic Regression
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Applying Softmax Activation function
def softmax(logits):
    exponentials = np.exp(logits)
    return exponentials / np.sum(exponentials, axis=1).reshape(-1, 1)

# Computing Loss over using logistic regression
def loss(Y, y_hat):
    return -np.sum(Y * np.log(y_hat)) / Y.shape[0]

# Adding bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

# Basically doing prediction but named forward as its 
# performing Forward-Propagation
def forward(X, w1, w2):
    h = sigmoid(np.matmul(prepend_bias(X), w1))
    y_hat = softmax(np.matmul(prepend_bias(h), w2))
    return y_hat

# Calling the predict() function
def classify(X, w1, w2):
    y_hat = forward(X, w1, w2)
    labels = np.argmax(y_hat, axis=1)
    return labels.reshape(-1, 1)

# Printing results to the terminal screen
def report(iteration, X_train, Y_train, X_test, Y_test, w1, w2):
    y_hat = forward(X_train, w1, w2)
    training_loss = loss(Y_train, y_hat)
    classifications = classify(X_test, w1, w2)
    accuracy = np.average(classifications == Y_test) * 100.0
    print("Iteration: %5d, Loss: %.6f, Accuracy: %.2f%%" %
          (iteration, training_loss, accuracy))