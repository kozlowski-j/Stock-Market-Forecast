# -*- coding: utf-8 -*-
from app_dash.utils import *
from app_dash.utils_tensorflow import *

from tensorflow import keras
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from app import app
from layout import main_layout
import callbacks


ticker = '^GSPC'
df = load_ticker_data(ticker, update=True, start_history='2012-01-01')
data = df['adjclose']

model = keras.models.load_model('../models/keras_tuned_model.h5')
past_history = 30
target_size = 7

dataset, _, col_scaler = prepare_dataset(df, 'adjclose')

LAST_SEQUENCE = dataset.shape[0] - 1 - past_history
test_batch = prepare_test_batch(dataset, LAST_SEQUENCE, None,  past_history).take(1)
prediction = model.predict(test_batch)[0]
prediction_rescaled = return_original_scale(prediction, col_scaler['adjclose'])

business_days = CustomBusinessDay(calendar=USFederalHolidayCalendar())
try:
    start_date = datetime.strptime(df.index[-1], '%Y-%m-%d') + timedelta(days=1)
except TypeError:
    start_date = df.index[-1] + timedelta(days=1)

prediction_index = pd.date_range(start=start_date, periods=target_size, freq=business_days)

data.index = pd.DatetimeIndex(data.index)


app.layout = main_layout


if __name__ == '__main__':
    app.run_server(debug=True)
