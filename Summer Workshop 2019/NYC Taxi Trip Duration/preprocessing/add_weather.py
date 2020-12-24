import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans


train = pd.read_csv('C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data/train.csv')
test = pd.read_csv('C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data/test.csv')

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

#天气数据
weather_hour = pd.read_csv('C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/external_data/weather.csv')

'''
weather = pd.read_csv('C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/external_data/weather_data_nyc_centralpark_2016.csv')
# Weather dataframe
weather.replace('T', 0.001, inplace=True)
weather['pickup_date'] = pd.to_datetime(weather['date'], dayfirst=True).dt.date
weather['average temperature'] = weather['average temperature'].astype(np.int64)
weather['precipitation'] = weather['precipitation'].astype(np.float64)
weather['snow fall'] = weather['snow fall'].astype(np.float64)
weather['snow depth'] = weather['snow depth'].astype(np.float64)
'''

# Weather hourly dataframe
weather_hour['Datetime'] = pd.to_datetime(weather_hour['pickup_datetime'], dayfirst=True)
weather_hour['pickup_date'] = weather_hour.Datetime.dt.date
weather_hour['hour'] = weather_hour['Datetime'].dt.hour
weather_hour['fog'] = weather_hour.fog.astype(np.int8)
weather_hour.drop(['pickup_datetime', 'Datetime'], inplace=True, axis=1)


#weather_hour去重 keep = 'first' or 'last'
weather_hour = weather_hour.drop_duplicates(subset=['pickup_date','hour'],keep='first')

#only pick these columns
#pressurm pressure in mBar
#hum humidity
#dewptm Dewpoint in Celcius
#tempm temperature in Celcius
#wspdm wind speed

weather_new = weather_hour[['pickup_date','hour','pressurem','hum','dewptm','tempm','wspdm']]

#把两个天气的数据加进去
#train = pd.merge(train, weather, on = 'pickup_date')
#train = pd.merge(train, weather_hour, on = ['pickup_date', 'hour'])
#test = pd.merge(test, weather, on = 'pickup_date')
#test = pd.merge(test, weather_hour, on = ['pickup_date', 'hour'])

#缺失值处理
weather_new = weather_new.fillna(method='ffill') # 临近填充

train = pd.merge(train, weather_new, how = 'left', on = ['pickup_date','hour'])
train = train.fillna(method='ffill')
test = pd.merge(test, weather_new, how = 'left', on = ['pickup_date','hour'])
test = test.fillna(method='ffill')




#OSRM数据
fr1 = pd.read_csv('D:/project/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('D:/project/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('D:/project/fastest_routes_test.csv',
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')

#方向
def bearing_array(lat1, lng1, lat2, lng2):
    #AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
'''
#速度
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(
        lat * 0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5)**2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


train.loc[:, 'distance_haversine'] = haversine_array(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train[
    'trip_duration']

#速度最快最慢的几个cluster和几天###
cluster_speed = train.loc[:,['avg_speed_h','pickup_cluster']]
cluster_speed_sort = cluster_speed.sort_values('avg_speed_h',inplace=False)

clusters = {}
for i in range(100):
    clusteri = cluster_speed.loc[cluster_speed['pickup_cluster']==i]
    avg_speed = clusteri['avg_speed_h'].sum()/len(clusteri)
    clusters[str(i)]=avg_speed
sorted_cluster = sorted(clusters.items(),key=lambda item:item[1])

fast_cluster = []
l = len(sorted_cluster)
for i in range(3):
    fast_cluster.append(sorted_cluster[l-1-i][0])

def isinlist(v):
    if v in fast_cluster:
        return 1
    else:
        return 0
    
train_is_fast = list(map(isinlist,cluster_speed.pickup_cluster.values))
train_is_fast_cluster = pd.DataFrame(train_is_fast,columns=['is_fast_cluster'])
train.loc[:,'is_fast_cluster'] = train_is_fast_cluster
test_is_fast = list(map(isinlist,cluster_speed.pickup_cluster.values))
test_is_fast_cluster = pd.DataFrame(test_is_fast,columns=['is_fast_cluster'])
test.loc[:,'is_fast_cluster'] = test_is_fast_cluster
'''



#去掉不用的变量
train.drop(['id', 'pickup_date', 'dropoff_datetime', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'dropoff_longitude', 'store_and_fwd_flag', 'trip_duration'], inplace=True, axis=1)
test.drop(['id', 'pickup_date', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_latitude', 'dropoff_longitude', 'store_and_fwd_flag'], inplace=True, axis=1)

#测试用
test.columns.values.tolist()
train.head()
