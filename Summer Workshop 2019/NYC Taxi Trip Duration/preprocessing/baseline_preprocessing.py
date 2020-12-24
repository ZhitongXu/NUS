import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
# KNN分类
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#KNN回归
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv('D:/project/train.csv')
test = pd.read_csv('D:/project/test.csv')

#限定地点边界
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85) 
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]

#限定时长边界
duration_border = (60, 10800)
train = train[train['trip_duration'] >= 60]
train = train[train['trip_duration'] <= 10800]

#从字符串改成日期类型
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime) 
##Not in Test

#月份
train['month'] = train['pickup_datetime'].dt.month
test['month'] = test['pickup_datetime'].dt.month
#星期几
train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
#几点
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour
#地点聚类
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

#字符转换成数字
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'])
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'])
train['store_and_fwd_flag_Y'] = store_and_fwd_flag_train['Y']
test['store_and_fwd_flag_Y'] = store_and_fwd_flag_test['Y']
train['store_and_fwd_flag_N'] = store_and_fwd_flag_train['N']
test['store_and_fwd_flag_N'] = store_and_fwd_flag_test['N']

#保留第一列id，输出submission用
test_id = test.iloc[:, 0]

#duration取log
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

#去掉不用的变量
train.drop(['id', 'pickup_date', 'dropoff_datetime', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'dropoff_longitude', 'store_and_fwd_flag', 'trip_duration'], inplace=True, axis=1)
test.drop(['id', 'pickup_date', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'dropoff_longitude', 'store_and_fwd_flag'], inplace=True, axis=1)


#重新排列，把duration放到最后一个
train = train[['vendor_id', 'passenger_count', 'store_and_fwd_flag_Y', 'store_and_fwd_flag_N', 'month', 'dayofweek', 'hour', 'pickup_cluster', 'dropoff_cluster', 'log_trip_duration']]
test = test[['vendor_id', 'passenger_count', 'store_and_fwd_flag_Y', 'store_and_fwd_flag_N', 'month', 'dayofweek', 'hour', 'pickup_cluster', 'dropoff_cluster']]


X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test = test
