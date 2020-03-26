# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import *
import pandas as pd
import plotly.graph_objects as go
from utils import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
data = pd.read_pickle('data/bpi_data.pkl')
data = update_bpi_file(data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index,
                         y=data.values,
                         name="BPI",
                         mode='lines',
                         line_color='deepskyblue'))

fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H2(children='Bitcoin Price Index Dashboard'),

    html.H3(show_current_date()),

    html.H5(f'Presented data range is: {data.index.min()} - {data.index.max()}'),

    html.Div(children='''
        Author: Kuba Kozłowski
    '''),

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


if __name__ == '__main__':
    app.run_server(debug=True)