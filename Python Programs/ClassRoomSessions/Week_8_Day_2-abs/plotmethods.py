import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

a= [1,2,3]
b=[4,5,6]
# x1,y1 = np.linspace(0,3,3), np.linspace(0,3,3)
# print(x1)
# print(y1)

x1_surf, y1_surf = np.meshgrid(a,b)
print(x1_surf , y1_surf)


x2_surf = x1_surf.ravel()
y2_surf = y1_surf.ravel()
print(x1_surf.ravel())
print(y1_surf.ravel())

print(x2_surf.reshape(x1_surf.shape))
print(y2_surf.reshape(y1_surf.shape))