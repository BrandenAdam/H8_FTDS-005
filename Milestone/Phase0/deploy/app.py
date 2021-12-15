# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:00:53 2021

@author: Shinsaragi
"""

import dash
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.FLATLY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
app.config.suppress_callback_exceptions = True