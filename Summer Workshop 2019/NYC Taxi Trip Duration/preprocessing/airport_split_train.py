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

train = train.drop(['id','store_and_fwd_flag','pickup_datetime','pickup_date','dropoff_datetime','trip_duration'],axis = 1)
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
train.loc[:, 'distance_to_jfk'] =  geo_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['jfk_lat'].values, train['jfk_lon'].values)
train.loc[:, 'distance_to_lg'] =  geo_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['lg_lat'].values, train['lg_lon'].values)

train = train.drop(['jfk_lat','jfk_lon','lg_lat','lg_lon'],axis=1)
jfk_trip_train = train[train['distance_to_jfk'] < 2]
lg_trip_train = train[train['distance_to_lg'] < 2]
train_airport = jfk_trip_train.append(lg_trip_train)
train_left = train[train['distance_to_jfk'] > 2]
train_left = train_left[train_left['distance_to_lg'] > 2]

train_airport = train_airport.drop(['distance_to_jfk','distance_to_lg'],axis=1)
train_left = train_left.drop(['distance_to_jfk','distance_to_lg'],axis=1)

test['jfk_lat'] = 40.639722
test['jfk_lon'] = -73.778889
test['lg_lat'] = 40.77725
test['lg_lon'] = -73.872611
test.loc[:, 'distance_to_jfk'] =  geo_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['jfk_lat'].values, test['jfk_lon'].values)
test.loc[:, 'distance_to_lg'] =  geo_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['lg_lat'].values, test['lg_lon'].values)

test = test.drop(['jfk_lat','jfk_lon','lg_lat','lg_lon'],axis=1)
jfk_trip_test = test[test['distance_to_jfk'] < 2]
lg_trip_test = test[test['distance_to_lg'] < 2]
test_airport = jfk_trip_test.append(lg_trip_test)
test_left = test[test['distance_to_jfk'] > 2]
test_left = test_left[test_left['distance_to_lg'] > 2]

test_airport = test_airport.drop(['distance_to_jfk','distance_to_lg'],axis=1)
test_left = test_left.drop(['distance_to_jfk','distance_to_lg'],axis=1)






# airport_train performance
# Try LightGBM with sklearn API
X_airport = train_airport.drop(['log_trip_duration'], axis=1)
y_airport = train_airport['log_trip_duration']

lgbm = lgb.LGBMRegressor(n_estimators=500, num_leaves=1000, max_depth=25, objective='regression')
lgbm.fit(X_airport, y_airport)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#from sklearn.metrics import mean_squared_error as MSE

#print(lgbm.score(X_train, y_train), lgbm.score(X_test, y_test))
#print(np.sqrt(MSE(y_test, lgbm.predict(X_test))))

#output
#print(lgbm.score(X_train, y_train), lgbm.score(X_test, y_test))
#0.8177820213397602 0.807571373976458
#print(np.sqrt(MSE(y_test, lgbm.predict(X_test))))
#0.23793652940699217


'''
# train_left performance
# Try LightGBM with sklearn API
X = train_left.drop(['log_trip_duration'], axis=1)
y = train_left['log_trip_duration']

lgbm = lgb.LGBMRegressor()
lgbm.fit(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_squared_error as MSE

print(lgbm.score(X_train, y_train), lgbm.score(X_test, y_test))
print(np.sqrt(MSE(y_test, lgbm.predict(X_test))))

#output
#print(lgbm.score(X_train, y_train), lgbm.score(X_test, y_test))
#0.6723239739023947 0.6731493631163679
#print(np.sqrt(MSE(y_test, lgbm.predict(X_test))))
#0.39490257066705264
'''

#Try RandomForest
X_left = train_left.drop(['log_trip_duration'], axis=1)
y_left = train_left['log_trip_duration']

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_left, y_left, test_size=0.2, random_state=42)

#from sklearn.metrics import mean_squared_error as MSE

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_left, y_left)

#print(rf.score(X_train, y_train), rf.score(X_test, y_test))
#print(np.sqrt(MSE(y_test, rf.predict(X_test))))

#output
#print(rf.score(X_train, y_train), rf.score(X_test, y_test))
#0.9561099487483097 0.7599263477750072
#print(np.sqrt(MSE(y_test, rf.predict(X_test))))
#0.3384445638414398

# Note: airport_train Try LightGBM with sklearn API has a better performance
#       train_left Try LightGBM with sklearn API is not so good but Try RandomForest has a similar performance
#       using only latitude and longitude 
#       as when using LightGBM cluster produced by kmeans is not as good as the raw.

train['is_jfk'] = (train['distance_to_jfk'] < 2).astype('int')
train['is_lg'] = (train['distance_to_lg'] < 2).astype('int')
test['is_jfk'] = (test['distance_to_jfk'] < 2).astype('int')
test['is_lg'] = (test['distance_to_lg'] < 2).astype('int')


#Make predictions on test data frame
test_airport['prediction'] = lgbm.predict(test_airport.drop(['id'],axis=1))
test_left['prediction'] = rf.predict(test_left.drop(['id'],axis=1))
test_res = pd.concat([test_airport,test_left],axis=0)
test_res['trip_duration'] = np.exp(test_res['prediction'].values) - 1

#Create a data frame designed a submission on Kaggle
submission = test_res[['id','trip_duration']]
submission.head()

#Create a csv out of the submission data frame
submission.to_csv("sub.csv", index=False)