# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import *
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs.layout import yaxis

from utils import *
from forecast import fit_arma

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
data = pd.read_pickle('data/bpi_data.pkl')
data = update_bpi_file(data)
p = 12
q = 0
data.index = pd.DatetimeIndex(data.index, freq='D')
pred_start = data.index.max()
pred_end = pred_start + pd.Timedelta(days=30)
pred = fit_arma(data, p, q, pred_start, pred_end)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index,
                         y=data.values,
                         name="BPI",
                         mode='lines',
                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=pred.index,
                         y=pred.values,
                         name="BPI forecast",
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


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H2(children='Bitcoin Price Index Dashboard'),

    html.H3(show_current_date()),

    html.H5(f'Presented data range is: {data.index.min()} - {data.index.max()}'),

    html.Div(
        id='div1'
    ),

    dcc.Graph(
        id='bpi-graph',
        figure=fig
    )

])


# @app.callback(
#     Output('bpi-graph', 'figure'),
#     [Input('datetime_RangeSlider', 'value')])
# def update_output(value):
#
#     start_date = '2020-03-0'+str(value[0])
#     end_date = '2020-03-'+str(value[1])
#
#     data = get_bpi(start_date, end_date)
#
#     figure = {
#         'data': [
#             {'x': data.index,
#              'y': data.values,
#              'type': 'line',
#              'name': 'BPI'},
#         ],
#         'layout': {
#             'title': 'BPI in March 2020'
#         }
#     }
#     return figure

##get this callback working. See https://dash.plot.ly/getting-started-part-2
@app.callback(
    Output('div1', 'children'),
    [Input('bpi-graph', 'figure')])
def get_ranges(figure):
    # Make data available in the scope of this function.
    global data

    # slider_range = figure.layout.xaxis.Rangeslider.range
    slider_range = figure['layout']['xaxis']['range']
    print(slider_range)
    output = f'''
            Author: Kuba Kozłowski
            range {slider_range}
        '''
    # new_y_low = data[slider_range[0]]
    # new_y_high = data[slider_range[1]]

    # fig.update_layout(
    #     yaxis=dict(
    #         autorange=False,
    #         range=[new_y_low, new_y_high]
    #     )
    # )

    return output


if __name__ == '__main__':
    app.run_server(debug=True)