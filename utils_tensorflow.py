import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from datetime import datetime, timedelta


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
        end_index = len(dataset)

    # print(start_index, end_index)
    for i in range(start_index, end_index):
        indices = range(i+1 - history_size, i+1)

        # Reshape data from (history_size,) to (history_size, 1)
        history_dates.append(np.reshape(dataset[indices], (history_size, 1)))

        if target_size == 1:
            future_dates.append(dataset[i + target_size])
        elif end_index == len(dataset):
            from pandas.tseries.holiday import USFederalHolidayCalendar
            from pandas.tseries.offsets import CustomBusinessDay

            business_days = CustomBusinessDay(calendar=USFederalHolidayCalendar())
            start_date = datetime.strptime(dataset[-1], '%Y-%m-%d') + timedelta(days=1)
            future_dates_list = pd.date_range(start=start_date, periods=target_size, freq=business_days)
            future_dates_list = np.array(future_dates_list.date, 'str')
            future_dates.append(future_dates_list)
        else:
            future_dates.append(dataset[i: i + target_size])

    dates_tensors = tf.data.Dataset.from_tensor_slices((history_dates, future_dates))
    batched_dates = dates_tensors.batch(batch_size).repeat()

    return batched_dates


def prepare_test_batch(dataset, start_index, end_index, history_size,
                       batch_size=64, buffer_size=1000):
    data = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)

    for i in range(start_index, end_index):
        indices = range(i + 1 - history_size, i + 1)
        data.append(dataset[indices])

    data = np.array(data)

    data_tensors = tf.data.Dataset.from_tensor_slices(data)
    batched_tensors = data_tensors.cache().shuffle(buffer_size).batch(batch_size).repeat()

    return batched_tensors


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


def multi_step_plot_dates(x_dates, history, y_dates, true_future=None, prediction=None):

    x_dates = [i.decode("utf-8") for i in x_dates.flatten()]
    x_dates = pd.DatetimeIndex(x_dates)

    y_dates = [i.decode("utf-8") for i in y_dates.flatten()]
    y_dates = pd.DatetimeIndex(y_dates)

    print(f'History date range: {x_dates[0]}, {x_dates[-1]}')
    print(f'Future date range: {y_dates[0]}, {y_dates[-1]}')

    plt.plot(x_dates, history, label='History')

    if true_future is not None:
        plt.plot(y_dates, true_future, 'b-o',
                 label='True Future')
    if prediction is not None:
        plt.plot(y_dates, prediction, 'r--o',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.show()


def return_original_scale(data, scaler):

    data_tmp = deepcopy(data)
    data_tmp = np.array(data_tmp)

    try:
        data_rescaled = scaler.inverse_transform(data_tmp)
    except ValueError or AttributeError:
        data_tmp = data_tmp.reshape(-1, 1)
        data_rescaled = scaler.inverse_transform(data_tmp)

    return data_rescaled

