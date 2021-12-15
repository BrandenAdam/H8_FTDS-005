# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:04:13 2021

@author: Shinsaragi
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from apps import home, hypo
from app import app
from app import server

# layout for home
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("Supermarket Sales Hacktiv8 Batch05 DASHBOARD",
                className="text-center"),
                className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(
                html.H5(children='My name is Branden, This is my milestone phase 00 dashboard'),
                className="mb-4")
        ]),

        dbc.Row([
            dbc.Col(
                html.H5(children='It consists of two main pages: Visualization, which gives a visualization of the graph of the data, '
                'Hypothesis, where my hypothesis testing and conclusion is written'),
                className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    children=[
                        html.H3(children='Get the original dataset here',
                        className="text-center"),
                        dbc.Button("Supermarket Sales Dataset",
                        href="https://www.kaggle.com/aungpyaeap/supermarket-sales?select=supermarket_sales+-+Sheet1.csv",
                        color="primary",
                        className="mt-3"),
                    ],
                    body=True, color="dark", outline=True
                ),
                width=6, className="mb-6",
            ),

            dbc.Col(
                dbc.Card(
                    children=[
                        html.H3(children='Hypothesis Testing',
                        className="text-center"),
                        dbc.Button("Hypothesis",
                        href="/apps/hypo",
                        color="primary",
                        className="mt-3",
                        ),
                    ], 
                    body=True, color="dark", outline=True 
                ),
                width=6, className="mb-6",
            ),
        ], justify="center", className="mb-5"),
    ])

])