import base64
import requests
import dash
import dash_core_components as dcc
import dash_html_components as html
from requests.auth import HTTPBasicAuth

app = dash.Dash()

app.layout = html.Div(
    children=[
        html.Tr(
            children=[
                html.Td(
                    dcc.Textarea(
                        id='input_data',
                        placeholder='Enter a post here',
                        value='',
                        rows=30,
                        cols=80,
                    )
                ),
                html.Td(
                    dcc.Textarea(
                        id='results',
                        value='',
                        rows=30,
                        cols=80,
                    )
                )
            ]
        ),
        html.Tr(
            html.Td(
                html.Button('Submit', id='button')
            )
        )
    ]
)

@app.callback(
    dash.dependencies.Output('results', 'value'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input_data', 'value')]
)
def display_results(n_clicks, value):
    r = requests.post(
        'http://localhost:8888/code',
        data={'rows': value},
        auth=HTTPBasicAuth('Admin', 'Admin'))
    return r.text

if __name__ == '__main__':
    app.run_server(debug=True)
