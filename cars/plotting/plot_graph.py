import matplotlib.pyplot as plt


# Plot the given model via history object
def plot_model(history, metric, name, save_location):
    # Plot training & validation accuracy values
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(name + 'Model ' + metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig(save_location)

