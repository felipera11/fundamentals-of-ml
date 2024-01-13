import numpy as np

def back(X, Y, y_hat, w2, h):
  w2_gradient = np.matmul(prepend_bias(h).T, (y_hat - Y)) / X.shape[0]
  w1_gradient = np.matmul(prepend_bias(X).T, np.matmul(y_hat - Y, w2[1:].T)
                                        * sigmoid_gradient(h)) / X.shape[0]
return (w1_gradient, w2_gradient)