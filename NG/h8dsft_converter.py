# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:54:52 2021

@author: Shinsaragi
"""

def c2k(c):                    #celcius to kelvin function
    return c + 273.15          #return result of celcius to kelvin conversion

def k2c(k):                     #kelvin to celcius function
    return k - 273.15           #return result of kelvin to celcius conversion

def t2f(t, types):                      #temperature to fahrenheit function
    if types == "celcius":              #check if types is celcius
        return (t * 9/5) + 32           #return result of celcius to fahrenheit conversion
    elif types == "kelvin":             #check if types is kelvin
        return (t - 273.15) * 9/5 + 32  #return result of kelvin to fahrenheit conversion
    
def f2t(f, types):                      #fahrenheit to temperature function
    if types == "celcius":              #check if types is celcius
        return (f - 32) * 5/9           #return result of fahrenheit to celcius conversion
    elif types == "kelvin":             #check if types is kelvin
        return (f - 32) * 5/9 + 273.15  #return result of fahrenheit to kelvin conversion





