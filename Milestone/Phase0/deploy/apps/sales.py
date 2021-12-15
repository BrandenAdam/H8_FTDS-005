# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:03:57 2021

@author: Shinsaragi
"""

import plotly.express as px
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app #change this line

# Data Loading
df = pd.read_csv("./supermarket_sales.csv")

# Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'],).dt.strftime('%m-%d-%Y')
df = df.drop(columns=["Invoice ID",])

select = ["Payment method", "Customer stratification rating", "Quantity of each sold product line", "Weekly total gross income"]

# layout for the visualization
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("Supermarket Sales Visualization"),
                className="mb-2 mt-2"
            )
        ]),
        dbc.Row([
            dbc.Col(
                html.H6(children='Visualizing the trends across the different stages of the Supermarket Sales'),
                className="mb-4"
            )
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='selected_graph',
                    options=[
                       {'label': opt, 'value': opt} for opt in select
                    ],
                    value='Payment method',
                ),
                className="mb-4"
            )
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='main-graph')
            ])
        ]),
         dbc.Row([
            dbc.Col([
                html.H3(["Visualization Exploration Analysis"]),
                html.P([
                    "- Customers use Ewallet and Cash more than Credit Card", 
                    html.Br(),
                    "- Branch B seems to be underperformed",
                    html.Br(),
                    "- Male seems to buy Health and beauty product more and Female seems to buy Fashion accessories product more",
                    html.Br(),
                    "- Around 24 February 2019, there seems to be less customer than usual",
                    html.Br(),
                    "- City and Product line are probably not related with each others"
                ]),
            ],
            className="mb-4")
        ]),
    ])
])
@app.callback(
    Output('main-graph', 'figure'),
    Input('selected_graph', 'value')
)
def update_visual_chart(select):
    if select == "Payment method":
        # make the pie chart then return it
        fig = px.pie(df, names='Payment', height=600)
        fig.update_traces(textposition='inside', textinfo='percent+label', title="Payment method")
        return fig

    elif select == "Customer stratification rating":
        # make the box chart then return it
        fig = px.box(df, x="Branch", y="Rating", color="Gender" , points = "all",
                title="Customer stratification rating")
        return fig

    elif select == "Quantity of each sold product line":
        # make the histogram then return it with
        fig = px.histogram(df, x="Product line", y="Quantity", color="Gender" , barmode="group", barnorm=None, 
                    title="Quantity of each sold product line")
        return fig

    elif select == "Weekly total gross income":
        # make new df and set index to date
        fig1 = df[["Date", "gross income", "Product line"]].set_index("Date")
        # pivot the table
        fig1 = fig1.pivot_table(values="gross income", index = fig1.index, columns="Product line", aggfunc='sum', fill_value=0)
        # make new column for  total gross income
        fig1["Total_gross_income"] = fig1.sum(axis=1)
        # sort index
        fig1 = fig1.sort_index(ascending=True, )
        # turn index to datetime so we can use it for resample
        fig1.index = pd.to_datetime(fig1.index)
        # resample the data weekly with mean calculation
        weekly = fig1.resample('W').mean()
        # make the line visualization then return it
        fig = px.line(weekly,  line_shape = "spline", 
                    title="Weekly total gross income(in usd)")
        return fig



    
