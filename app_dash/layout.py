import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from utils import show_current_date, get_tickers_dict


main_layout = html.Div(children=[
    html.H2(children='S&P 500 Index Dashboard'),

    html.H4(show_current_date()),

    html.P('This dashboard is my learning effort during Machine Learning bootcamp. '),
    html.P('The ultimate purpose of creating it is learning prediction of time series data.'),

    html.Div(
        dcc.Dropdown(
            id='ticker-dropdown',
            options=get_tickers_dict(),
            value='^GSPC'
        ),
        style=dict(
            width=200
        )
    ),

    dcc.Graph(
        id='main-graph'
    )],

    style=dict(
        marginLeft='20%',
        marginRight='20%',
        marginTop='1%'
    )
)
