# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:01:14 2021

@author: Shinsaragi
"""

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from app import server

from apps import home, sales, hypo

# layout for index
app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Explore Data", href='/apps/sales')),
        ],
        brand="Batch05 Branden Dashboard",
        brand_href="/apps/home",
        color="dark",
        dark=True,
        sticky='top'
    ),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=[])
])
@app.callback(
    Output(component_id='page-content', component_property='children'),
    [Input(component_id='url', component_property='pathname')])
def display_page(pathname):
    if pathname == '/apps/sales':
        return sales.layout
    elif pathname == '/apps/hypo':
        return hypo.layout
    else:
        return home.layout


if __name__ == '__main__':
    app.run_server(debug=True)