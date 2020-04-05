import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt


# from ts_utils import check_time_series_stationary
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


def mse(y, yhat):
    return np.mean((y - yhat) ** 2)


def mae(y, yhat):
    return np.median(np.abs(y - yhat))


def fit_arma(ts, p, q, pred_start, pred_end):
    arma = ARIMA(ts, order=(p, 0, q)).fit()
    pred = arma.predict(start=pred_start, end=pred_end, dynamic=True)

    return pred

    # print(f'Ar params {arma.arparams}')
    # print(f'Ar params {arma.maparams}')
    # print(f'AIC: {arma.aic}, BIC: {arma.bic}')

    # dw_test = sm.stats.durbin_watson(arma.resid)
    # print(f"Statystyka Durbina-Watsona {dw_test}")

    # normal_test = stats.shapiro(arma.resid)
    # print(f"Pvalue testu Shapiro-Wilka: {normal_test[1]}")

    # pred_size = pred.values.size
    # mse_val = mse(ts.values[-pred_size:], pred.values)
    # mae_val = mae(ts.values[-pred_size:], pred.values)
    # print(f'MSE: {mse_val}')
    # print(f'MAE: {mae_val}')

    # plt.figure(figsize=(16, 6))
    # plt.plot(ts)
    # plt.plot(pred, 'g--')
    # plt.show()


def find_the_best_arma(ts, max_p=10, max_q=10, ar=False):
    aic_dic, bic_dic = {}, {}

    for p in np.arange(1, max_p + 1):
        for q in np.arange(1, max_q + 1):
            arma = ARIMA(ts, order=(p, 0, q)).fit()
            aic_dic[f'ARMA({p}, {q})'] = arma.aic
            bic_dic[f'ARMA({p}, {q})'] = arma.bic
            # print(p, q, arma.aic, arma.bic)

    print(min(aic_dic, key=aic_dic.get))
    print(min(bic_dic, key=bic_dic.get))

#################################
#  TEST
#################################
# data = pd.read_pickle('data/bpi_data.pkl')
# data.index = pd.DatetimeIndex(data.index, freq='D')
# ts = data
# p = 12
# q = 2
# pred_start = '2020-04-01'
# pred_end = '2020-04-30'

# fit_arma(ts, p, q, pred_start, pred_end)