import datetime
import pandas as pd


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
