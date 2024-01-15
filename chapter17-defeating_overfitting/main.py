# A regularized four-layered neural network.

# importing the necessary modules
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l1
import echidna as data
import losses
import boundary

# Data pre-processing
X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)
Y_validation = to_categorical(data.Y_validation)

# Building neural network model
model = Sequential()
model.add(Dense(100, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(30, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(2, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Calling the training function for our neural network
history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=2000, batch_size=25)

print("\n\nDecision boundaries in making ... \n\n\n")


boundary.show(model, data.X_train, data.Y_train,
              title="Training-set")
boundary.show(model, data.X_validation, data.Y_validation,
              title="Validation-set")
losses.plot(history)
