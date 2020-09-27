from dash.dependencies import Input, Output

from app import app


# TODO: get this callback working. See https://dash.plot.ly/getting-started-part-2
@app.callback(
    Output('div1', 'children'),
    [Input('bpi-graph', 'figure')])
def get_ranges(figure):
    # Make data available in the scope of this function.
    global data

    # slider_range = figure.layout.xaxis.Rangeslider.range
    slider_range = figure['layout']['xaxis']['range']
    # print(slider_range)
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