import os
import pandas as pd
import numpy as np
from math import sqrt
import lightgbm as lgb

os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train = train[train['trip_duration'] > 60]
train = train[train['trip_duration'] <= 10800]

train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

# Extracting Month, day of Week and Hour
train['Month'] = train['pickup_datetime'].dt.month
test['Month'] = test['pickup_datetime'].dt.month

train['DayofMonth'] = train['pickup_datetime'].dt.day
test['DayofMonth'] = test['pickup_datetime'].dt.day

train['Hour'] = train['pickup_datetime'].dt.hour
test['Hour'] = test['pickup_datetime'].dt.hour

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek

# exclude dots that are outside of New York
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]

train = train.drop(['store_and_fwd_flag','pickup_datetime','pickup_date','dropoff_datetime','trip_duration'],axis = 1)
test = test.drop(['store_and_fwd_flag','pickup_datetime','pickup_date'], axis = 1)

# helper function
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def geo_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


train['jfk_lat'] = 40.639722
train['jfk_lon'] = -73.778889
train['lg_lat'] = 40.77725
train['lg_lon'] = -73.872611
train.loc[:, 'pickup_distance_to_jfk'] =  geo_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['jfk_lat'].values, train['jfk_lon'].values)
train.loc[:, 'pickup_distance_to_lg'] =  geo_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['lg_lat'].values, train['lg_lon'].values)
train.loc[:, 'dropoff_distance_to_jfk'] =  geo_distance(train['dropoff_latitude'].values, train['dropoff_longitude'].values, train['jfk_lat'].values, train['jfk_lon'].values)
train.loc[:, 'dropoff_distance_to_lg'] =  geo_distance(train['dropoff_latitude'].values, train['dropoff_longitude'].values, train['lg_lat'].values, train['lg_lon'].values)

def jfk_trip(x):
    return int(x['pickup_distance_to_jfk'] < 2 or x['dropoff_distance_to_jfk'] < 2)

def lg_trip(x):
    return int(x['pickup_distance_to_lg'] < 2 or x['dropoff_distance_to_lg'] < 2)

train = train.drop(['jfk_lat','jfk_lon','lg_lat','lg_lon'],axis=1)
train['is_jfk'] = train.apply(jfk_trip,axis=1)
train['is_lg'] = train.apply(lg_trip,axis=1)

test['jfk_lat'] = 40.639722
test['jfk_lon'] = -73.778889
test['lg_lat'] = 40.77725
test['lg_lon'] = -73.872611
test.loc[:, 'pickup_distance_to_jfk'] = geo_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['jfk_lat'].values, test['jfk_lon'].values)
test.loc[:, 'pickup_distance_to_lg'] = geo_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['lg_lat'].values, test['lg_lon'].values)
test.loc[:, 'dropoff_distance_to_jfk'] = geo_distance(test['dropoff_latitude'].values, test['dropoff_longitude'].values, test['jfk_lat'].values, test['jfk_lon'].values)
test.loc[:, 'dropoff_distance_to_lg'] = geo_distance(test['dropoff_latitude'].values, test['dropoff_longitude'].values, test['lg_lat'].values, test['lg_lon'].values)

test = test.drop(['jfk_lat','jfk_lon','lg_lat','lg_lon'],axis=1)
test['is_jfk'] = test.apply(jfk_trip,axis=1)
test['is_lg'] = test.apply(lg_trip,axis=1)

train = train.drop(['pickup_distance_to_jfk','pickup_distance_to_lg','dropoff_distance_to_jfk','dropoff_distance_to_lg'],axis=1)
test = test.drop(['pickup_distance_to_jfk','pickup_distance_to_lg','dropoff_distance_to_jfk','dropoff_distance_to_lg'],axis=1)

airport_train = train[['id','is_jfk','is_lg']]
airport_test = test[['id','is_jfk','is_lg']]
airport_test.to_csv("airport_test.csv", index=False)
airport_train.to_csv("airport_train.csv", index=False)

'''
train['is_jfk'] = (train['distance_to_jfk'] < 2).astype('int')
train['is_lg'] = (train['distance_to_lg'] < 2).astype('int')
test['is_jfk'] = (test['distance_to_jfk'] < 2).astype('int')
test['is_lg'] = (test['distance_to_lg'] < 2).astype('int')
'''