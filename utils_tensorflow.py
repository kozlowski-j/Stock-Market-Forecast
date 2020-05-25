import numpy as np
import matplotlib.pyplot as plt


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


def prepare_dates(dataset, start_index, end_index, history_size, target_size):
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

    return np.array(history_dates), np.array(future_dates)


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


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i: i+target_size])

    return np.array(data), np.array(labels)


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


def multi_step_plot_dates(time_index, history, true_future, prediction):
    #     num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(time_index.flatten(), history[:, 1], label='History')
    plt.plot(np.arange(num_out), true_future, 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), prediction, 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.show()


