import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")
train = pd.read_csv('his.csv')

db = pd.DataFrame
import plotly.express as px
fig = px.histogram(train['route'], x="total_bill")
fig.show()