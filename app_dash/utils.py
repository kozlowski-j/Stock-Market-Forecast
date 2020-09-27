import datetime
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
import os
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def get_bpi(start_date, end_date):
    """

    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :return: pandas Series with BPI, and dates as index
    """
    api_link = "https://api.coindesk.com/v1/bpi/historical/close.json?start="
    api_query = api_link + start_date + "&end=" + end_date

    # Save BPI into pandas Series
    bpi_series = pd.read_json(api_query).bpi

    # Drop technical fields (last two rows)
    bpi_series.drop(['updated', 'updatedISO'], inplace=True)

    return bpi_series


def update_bpi_file(data):
    # Get date of last record in current data
    last_date = data.index[-1]
    last_date = datetime.datetime(int(last_date[:4]),   # year
                                  int(last_date[5:7]),  # month
                                  int(last_date[8:]))   # day

    # Add one day, to omit overlapping values
    last_date_plus_one = last_date + datetime.timedelta(days=1)
    last_date_plus_one_as_text = last_date_plus_one.strftime('%Y-%m-%d')

    # Get date of last finished calendar day (yesterday)
    today_date = datetime.datetime.now()
    yesterday_date = today_date - datetime.timedelta(days=1)
    yesterday_date_as_text = yesterday_date.strftime('%Y-%m-%d')

    # Check if data isn't already update
    if last_date_plus_one_as_text == today_date.strftime('%Y-%m-%d'):
        return data

    # Get data from period between last record and yesterday
    new_bpi = get_bpi(last_date_plus_one_as_text, yesterday_date_as_text)
    data = data.append(new_bpi)

    # Update data file (overwrite)
    data.to_pickle('data/bpi_data.pkl')

    return data


def show_current_date():
    return 'Today is: ' + str(datetime.datetime.now().strftime('%Y-%m-%d'))


def load_ticker_data(ticker, data_path='../data', update=False,
                     start_history=None, end_history=None):

    data = None

    try:
        for file in os.listdir(data_path):
            if file.startswith(f"{ticker}"):
                data = pd.read_pickle(os.path.join(data_path, file))
                print(f'{ticker} data imported from file.')
        if data is None:
            raise FileNotFoundError
    except FileNotFoundError:
        try:
            data = si.get_data(ticker, start_date=start_history, end_date=end_history)
            print(f'{ticker} data imported from API.')
        except (KeyError, AssertionError):
            return pd.DataFrame()

    if update:
        last_day = data.index.max()
        if last_day < datetime.datetime.now():
            new_data = si.get_data(ticker, start_date=last_day)
            data = data.append(new_data)

    # TODO: Save file only if there is new data
    data.to_pickle(data_path + f'\\{ticker}.pkl')

    return data


def get_run_dir(folder_name='models'):
    import os
    import time
    root_dir = os.path.join(os.curdir, folder_name)
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_dir, run_id)


def prepare_dataset(data, target_variable):
    df = data.copy()
    history_start = '2012-01-01'
    df = df[df.index >= history_start]
    df.dropna(inplace=True)

    features_considered = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
    features = df[df['open'].isna() == False][features_considered]

    data_scaled = features.copy()

    TRAIN_SPLIT = int(df.shape[0] * 0.75)

    column_scaler = {}
    # scale the data (prices) from 0 to 1
    for column in data_scaled.columns:
        column_values = data_scaled[column].values.reshape(-1, 1)
        # Fit only on training data
        scaler = MinMaxScaler()
        scaler.fit(column_values[:TRAIN_SPLIT])
        data_scaled[column] = scaler.transform(column_values)
        column_scaler[column] = scaler

    target = data_scaled[target_variable].values
    dataset = data_scaled.values

    return dataset, target, column_scaler


def get_tickers_history(tickers, start_history, end_history=None):

    tickers_data = []
    maxlen = 0

    for ticker in tickers:
        try:
            t_data = load_ticker_data(ticker, update=False,
                                  start_history=start_history, end_history=end_history)
            t_data.drop(columns='ticker', inplace=True)
            # Add ticker suffix to columns' names.
            t_data.columns = [col+'_'+ticker for col in t_data.columns]
            # Pad sequences if they are shorter than longest
            # (for indexes which appeared on market after 'start_history').
            # maxlen = np.max([maxlen, t_data.shape[0]])
            # if maxlen > t_data.shape[0]:
            #     t_data = keras.preprocessing.sequence.pad_sequences(t_data, maxlen)
        except KeyError:
            t_data = pd.DataFrame()
        tickers_data.append(t_data)
        time.sleep(1)

    data_output = pd.concat(tickers_data, axis=1)

    return data_output


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
