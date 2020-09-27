import dash_core_components as dcc
import dash_html_components as html

from utils import show_current_date


main_layout = html.Div(children=[
    html.H2(children='S&P 500 Index Dashboard'),

    html.H4(show_current_date()),

    html.H5(children='This dashboard is my learning effort during Machine Learning bootcamp. '
                     'The ultimate purpose of creating it is learning prediction of time series data.'),

    # html.H5(f'Presented data range is: {data.index.min()} - {data.index.max()}'),

    html.Div(
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[
                {'label': '^GSPC', 'value': '^GSPC'}
            ],
            value='^GSPC'
        )

    ),

    dcc.Graph(
        id='main-graph'
    )

])
