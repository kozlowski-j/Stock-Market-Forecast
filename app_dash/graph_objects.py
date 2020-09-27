import plotly.graph_objects as go


def plot_ticker_ts(ticker, data, prediction):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index,
                             y=data.values,
                             name=ticker,
                             mode='lines',
                             line_color='deepskyblue'))

    fig.add_trace(go.Scatter(x=prediction_index,
                             y=prediction_rescaled.flatten(),
                             name="Tensorflow forecast",
                             mode='lines',
                             line_color='green',
                             line_dash='dash'))

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(step="all")
                ])
            ),
            # autorange=True,
            # rangeslider=dict(
            #     visible=True
            # ),
            type="date"
        ),
        yaxis=dict(
            autorange=True
        ),
        height=400,
        width=800
    )
    return fig
