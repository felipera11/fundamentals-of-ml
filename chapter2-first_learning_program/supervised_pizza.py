import numpy as np
#measure time
import time

initial_time = time.time()

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

def predict(X, w):
    return X * w

print(predict(20,2.1))
print(predict(14,1.5))

#error = predict(X,w) - Y
#squared_error = error ** 2

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

print(loss(X, Y, 1.5))

def train(X, Y, iterations, lr):
    w=0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
        
# Train the system
w = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f" % w)

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))

final_time = time.time()
print("Time taken: ", final_time - initial_time)

# Plot the chart
import matplotlib.pyplot as plt
plt.plot(X, Y, "bo")
plt.xlabel("Reservations")
plt.ylabel("Pizzas")
plt.axis([0, 50, 0, 50])
plt.plot([0, 50], [predict(0, w), predict(50, w)])
plt.show()

#adding a bias

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


# Import the dataset
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=10000, lr=0.01)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict the number of pizzas
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))

# Plot the chart
import matplotlib.pyplot as plt
plt.plot(X, Y, "bo")
plt.xlabel("Reservations")
plt.ylabel("Pizzas")
plt.axis([0, 50, 0, 50])
plt.plot([0, 50], [predict(0, w, b), predict(50, w, b)])
plt.show()