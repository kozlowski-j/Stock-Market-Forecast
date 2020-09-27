from datetime import datetime, timedelta
import pandas as pd
from dash.dependencies import Input, Output
from tensorflow import keras
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from app import app
from graph_objects import plot_ticker_ts
from utils import load_ticker_data, prepare_dataset
from utils_tensorflow import prepare_test_batch, return_original_scale


@app.callback(
    Output('main-graph', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def plot_main_graph(ticker):
    df = load_ticker_data(ticker, update=True, start_history='2012-01-01')
    data = df['adjclose']

    model = keras.models.load_model('../models/keras_tuned_model.h5')
    past_history = 30
    target_size = 7

    dataset, _, col_scaler = prepare_dataset(df, 'adjclose')

    LAST_SEQUENCE = dataset.shape[0] - 1 - past_history
    test_batch = prepare_test_batch(dataset, LAST_SEQUENCE, None, past_history).take(1)
    prediction = model.predict(test_batch)[0]
    prediction_rescaled = return_original_scale(prediction, col_scaler['adjclose'])

    business_days = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    try:
        start_date = datetime.strptime(df.index[-1], '%Y-%m-%d') + timedelta(days=1)
    except TypeError:
        start_date = df.index[-1] + timedelta(days=1)

    prediction_index = pd.date_range(start=start_date, periods=target_size, freq=business_days)
    prediction_final = pd.Series(data=prediction_rescaled.flatten(),
                                 index=prediction_index)

    data.index = pd.DatetimeIndex(data.index)

    fig = plot_ticker_ts(ticker, data, prediction_final)

    return fig


# # TODO: get this callback working. See https://dash.plot.ly/getting-started-part-2
# @app.callback(
#     Output('div1', 'children'),
#     [Input('bpi-graph', 'figure')])
# def get_ranges(figure):
#     # Make data available in the scope of this function.
#     global data
#
#     # slider_range = figure.layout.xaxis.Rangeslider.range
#     slider_range = figure['layout']['xaxis']['range']
#     # print(slider_range)
#     output = f'''
#             Author: Kuba Kozłowski
#             range {slider_range}
#         '''
#     # new_y_low = data[slider_range[0]]
#     # new_y_high = data[slider_range[1]]
#
#     # fig.update_layout(
#     #     yaxis=dict(
#     #         autorange=False,
#     #         range=[new_y_low, new_y_high]
#     #     )
#     # )
#
#     return output