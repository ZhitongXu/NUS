import os
import pandas as pd
import numpy as np
os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")

train = pd.read_csv('new_train.csv')
train_sub = train[:125000]

from sklearn.model_selection import train_test_split
train_xy, offline_test = train_test_split(train_sub,test_size = 0.2,random_state=21)

train_xy.to_csv("my_train.csv",index=False)
offline_test.to_csv("my_test.csv",index=False)

