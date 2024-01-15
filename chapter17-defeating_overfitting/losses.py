# A utility function that plots the training loss and validation loss from
# a Keras history object.

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def plot(history):
    plt.clf()
    plt.plot(history.history['loss'], label='Training set',
             color='blue', linestyle='-')
    plt.plot(history.history['val_loss'], label='Validation set',
             color='green', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(history.history['loss']))
    plt.legend()
    plt.show()