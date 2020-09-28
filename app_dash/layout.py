import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from utils import show_current_date, get_tickers_dict


main_layout = html.Div(children=[
    html.H2(children='S&P 500 Index Dashboard'),

    html.H4(show_current_date()),

    html.P(children='This dashboard is my learning effort during Machine Learning bootcamp. '
                     'The ultimate purpose of creating it is learning prediction of time series data.'),

    # html.H5(f'Presented data range is: {data.index.min()} - {data.index.max()}'),

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
        marginLeft='10%',
        marginRight='10%',
        marginTop='1%'
    )
)
