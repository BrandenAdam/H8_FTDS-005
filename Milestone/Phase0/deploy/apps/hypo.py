# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:04:13 2021

@author: Shinsaragi
"""

import plotly.express as px
import pandas as pd
from scipy import stats

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app

# Data Loading
df = pd.read_csv("./supermarket_sales.csv")

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'],).dt.strftime('%m-%d-%Y')
df = df.drop(columns=["Invoice ID",])

# make function for the hypothesis testing
def make_hypo():
    # make the contigency table, calculate it then check if it accept or reject null hypothesis
    contingency_table=pd.crosstab(df["City"],df["Product line"]),
    stat, p, dof, expected = stats.chi2_contingency(contingency_table)
    if p > 0.05:
        result = "Accept null hypothesis"
    else:
        result = "reject null hypothesis"

    # return it in html h5
    return html.H5(["stat=%.3f, p=%.3f" % (stat, p), 
                    html.Br(), 
                    result, 
                    html.Br(),
                    "City and Product line are probably not related with each others", 
                    ])

# layout for the hypothesis testing
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("Hypothesis Testing",
                className="text-center"),
                className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(
                html.H5(children='This is my Hypothesis Testing'),
                className="mb-4")
        ]),

        dbc.Row([
            dbc.Col(
                    html.P([
                        "Is the City and the Product line related?", 
                        html.Br(), 
                        "H0: City and Product line are not related",
                        html.Br(), 
                        "H1: City and Product line are related",
                        html.Br(),
                        "Jenis Hypothesis: Chi Square"
                    ]),
                
                className="mb-5")
        ]),
        
        dbc.Row([
            dbc.Col(
                # call the function
                make_hypo(),
                className="mb-4")
        ]),
        
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    # make the histogram and show the figure
                    figure=px.histogram(df, x="City", y="Quantity", color="Product line", barmode="group")
                )
            )
        ]),

       
    ])

])
    



