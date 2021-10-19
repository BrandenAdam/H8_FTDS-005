# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:09:40 2021

@author: Shinsaragi
"""

import random

rand = random.randint(1, 10)
flag = "y" 
count = 0
while flag == "y":
    if count >= 3: print("Game Over"); break
    count += 1
    inputs = int(input("Choose a number between 1 to 10: "))
    if inputs < rand: print("Guessed number is too small")
    elif inputs > rand: print("Guessed number is too big")
    else: print("Congratz"); count = 0; flag = input("Want to play again? (y/n)") 
    
    
    
    