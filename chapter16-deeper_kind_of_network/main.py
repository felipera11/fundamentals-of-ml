# importing the necessary modules
from keras.models import Sequential 
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical 
import echidna as data
import boundary

# Data pre-processing
X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)
Y_validation = to_categorical(data.Y_validation)

# Neural network building
model = Sequential()
model.add(Dense(100, activation='sigmoid')) 
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# Compiling the neural network for building our model
model.compile(loss='categorical_crossentropy', 
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Training phase of our neural netowrk model
model.fit(X_train, Y_train,
          validation_data=(X_validation, Y_validation),
          epochs=1000, batch_size=25)

# Displaying descesion boundary
#boundary.plot_boundary(model, data.X_train, data.Y_train)

import matplotlib.pyplot as plt

# Plotting the model architecture
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Model Architecture')
plt.plot(model.history.history['loss'], label='Training Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.title('Model Accuracy')
plt.plot(model.history.history['accuracy'], label='Training Accuracy')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


