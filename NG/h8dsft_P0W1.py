# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 10:52:29 2021

@author: Shinsaragi
"""

import numpy as np
import pandas as pd


data = pd.read_csv("./datasets/Pokemon.csv")
data.columns = ["index", "name", "type_1", "type_2", "total", "hp", "attack", "defense", 
                "sp_atk", "sp_def", "speed", "generation", "legendary"]

categorical = ["name", "type_1", "type_2", "legendary"]
numeric = [i for i in data if i not in categorical]
# change nan value of categorical data to None rather than just remove it and change nan value of numerical to median of the column
data[categorical] = data[categorical].fillna("None")
data[numeric] = data[numeric].fillna(data[numeric].median())

# data[categorical] = OrdinalEncoder().fit_transform(data[categorical])


