import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_time_steps(length):
    return list(range(-length, 0))


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])

    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


def show_plot_dates(time_steps, plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']

    if delta:
        future = time_steps[-delta:]
    else:
        future = time_steps[-1]

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps.flatten(), plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xticks(rotation=30)
    plt.xlabel('Time-Step')
    return plt


def prepare_batches(dataset, target, start_index, end_index, history_size,
                    target_size, batch_size=64, buffer_size=1000):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])

        if target_size == 1:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i: i+target_size])

    data = np.array(data)
    labels = np.array(labels)

    # Split data into tensors, according to the tutorial
    # Buffer and batch sizes acc. to the tutorial as well
    data_tensors = tf.data.Dataset.from_tensor_slices((data, labels))
    batched_tensors = data_tensors.cache().shuffle(buffer_size).batch(batch_size).repeat()

    return batched_tensors


def prepare_dates(dataset, start_index, end_index, history_size, target_size, batch_size=64):
    """
    This functions prepares date indexes for prediction and plotting.

    Args:
        dataset:
        start_index:
        end_index:
        history_size:
        target_size:
        batch_size:

    Returns:

    """
    history_dates = []
    future_dates = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)

        # Reshape data from (history_size,) to (history_size, 1)
        history_dates.append(np.reshape(dataset[indices], (history_size, 1)))

        if target_size == 1:
            future_dates.append(dataset[i + target_size])
        else:
            future_dates.append(dataset[i: i + target_size])

    dates_tensors = tf.data.Dataset.from_tensor_slices((history_dates, future_dates))
    batched_dates = dates_tensors.batch(batch_size).repeat()

    return batched_dates


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def multi_step_plot_dates(x_dates, history, y_dates, true_future, prediction=None):

    plt.plot(x_dates.flatten(), history[:, 1], label='History')
    plt.plot(y_dates.flatten(), true_future, 'b-o',
             label='True Future')
    if prediction:
        plt.plot(y_dates.flatten(), prediction, 'r-o',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.show()


