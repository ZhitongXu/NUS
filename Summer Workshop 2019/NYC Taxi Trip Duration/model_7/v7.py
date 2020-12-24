import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# excluding those less than 10s and more than 22h
train = train[(train.trip_duration < 5900)]
train = train[(train.passenger_count > 0)]

train = train[(train.pickup_longitude > -100)]
train = train[(train.pickup_latitude < 50)]

# transform regression target using log
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

# process store_and_fwd_flag (from Y/N to 1/0)
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

# process time
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

# extracting Month, dayofWeek and hour
train['Month'] = train['pickup_datetime'].dt.month
test['Month'] = test['pickup_datetime'].dt.month

train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour

train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek

# transform the date to the minute in a day
train['minute'] = train['pickup_datetime'].dt.minute
test['minute'] = test['pickup_datetime'].dt.minute
train['minute'] = train['hour'] + train['minute']/60
test['minute'] = test['hour'] + test['minute']/60

# add feature is_jfk and is_lg
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(
        lat * 0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5)**2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

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

# generate 100 clusters using kmeans
train.loc[:, 'distance_haversine'] = haversine_array(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(
    test['pickup_latitude'].values, test['pickup_longitude'].values,
    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:1000000]
kmeans = MiniBatchKMeans(n_clusters=100,
                         batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(
    train[['pickup_latitude', 'pickup_longitude']]).astype(object)
train.loc[:, 'dropoff_cluster'] = kmeans.predict(
    train[['dropoff_latitude', 'dropoff_longitude']]).astype(object)
test.loc[:, 'pickup_cluster'] = kmeans.predict(
    test[['pickup_latitude', 'pickup_longitude']]).astype(object)
test.loc[:, 'dropoff_cluster'] = kmeans.predict(
    test[['dropoff_latitude', 'dropoff_longitude']]).astype(object)

# add feature
group_freq = '60min'
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
train.loc[:, 'pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
test.loc[:, 'pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

# Count trips over 60min
df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']
train = train.merge(df_counts, on='id', how='left')
test = test.merge(df_counts, on='id', how='left')

# Count how many trips are going to each cluster over time
dropoff_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('dropoff_cluster').rolling('240min').mean() \
    .drop('dropoff_cluster', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})

train['dropoff_cluster_count'] = train[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)
test['dropoff_cluster_count'] = test[['pickup_datetime_group', 'dropoff_cluster']].merge(dropoff_counts, on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)

# Count how many trips are going from each cluster over time
df_all = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]
pickup_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby('pickup_cluster').rolling('240min').mean() \
    .drop('pickup_cluster', axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_cluster_count'})

train['pickup_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
test['pickup_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster']].merge(pickup_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)

# Count how many trips each route over time
pickup_dropoff_counts = df_all \
    .set_index('pickup_datetime') \
    .groupby([pd.TimeGrouper(group_freq), 'pickup_cluster','dropoff_cluster']) \
    .agg({'id': 'count'}) \
    .reset_index().set_index('pickup_datetime') \
    .groupby(['pickup_cluster','dropoff_cluster']).rolling('240min').mean() \
    .drop(['pickup_cluster','dropoff_cluster'], axis=1) \
    .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index() \
    .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'pickup_dropoff_cluster_count'})

train['pickup_dropoff_cluster_count'] = train[['pickup_datetime_group', 'pickup_cluster','dropoff_cluster']].merge(pickup_dropoff_counts, on=['pickup_datetime_group', 'pickup_cluster','dropoff_cluster'], how='left')['pickup_dropoff_cluster_count'].fillna(0)
test['pickup_dropoff_cluster_count'] = test[['pickup_datetime_group', 'pickup_cluster','dropoff_cluster']].merge(pickup_dropoff_counts, on=['pickup_datetime_group', 'pickup_cluster','dropoff_cluster'], how='left')['pickup_dropoff_cluster_count'].fillna(0)


cnt_pickup_dropoff = df_all.groupby(by=['pickup_cluster','dropoff_cluster'])
newdf = cnt_pickup_dropoff.size()
newdf = newdf.reset_index(name='cnt_pickup_dropoff')
train = pd.merge(train,newdf,how='left',on=['pickup_cluster','dropoff_cluster'])
test = pd.merge(test,newdf,how='left',on=['pickup_cluster','dropoff_cluster'])



coords_train = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))
coords_test = np.vstack((test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))
coords_join = np.vstack((coords_train,coords_test))
sample_ind = np.random.permutation(len(coords_join))[:1000000]
kmeans = MiniBatchKMeans(n_clusters=16,
                         batch_size=10000).fit(coords_join[sample_ind])


train.loc[:, 'pickup_district'] = kmeans.predict(
    train[['pickup_latitude', 'pickup_longitude']]).astype(object)
train.loc[:, 'dropoff_district'] = kmeans.predict(
    train[['dropoff_latitude', 'dropoff_longitude']]).astype(object)
test.loc[:, 'pickup_district'] = kmeans.predict(
    test[['pickup_latitude', 'pickup_longitude']]).astype(object)
test.loc[:, 'dropoff_district'] = kmeans.predict(
    test[['dropoff_latitude', 'dropoff_longitude']]).astype(object)


pickup_speeds = {}
dropoff_speeds = {}
clusters_speeds = {}
for pick_num in range(100):
    cluster_i = train[train.pickup_cluster == pick_num].copy()
    pickup_speeds[str(pick_num)] = cluster_i.avg_speed_h.values.mean()
    cluster_i_speeds = {}
    for sample in cluster_i.values:
        if str(sample[-1]) in cluster_i_speeds.keys():
            cluster_i_speeds[str(sample[-1])].append(
                sample[-3])  #-1代表dropoff_cluster在数据中的最后一列
        else:
            cluster_i_speeds[str(sample[-1])] = []
            cluster_i_speeds[str(sample[-1])].append(sample[-3])

    cluster_i_avg = {}
    for key in cluster_i_speeds.keys():
        cluster_i_avg[key] = np.mean(cluster_i_speeds[key])

    clusters_speeds[str(pick_num)] = cluster_i_avg

for drop_num in range(100):
    cluster_i = train[train.pickup_cluster == drop_num].copy()
    dropoff_speeds[str(drop_num)] = cluster_i.avg_speed_h.values.mean()

for p in range(100):
    if len(clusters_speeds[str(p)]) != 100:
        for d in range(100):
            if str(d) not in clusters_speeds[str(p)].keys():
                clusters_speeds[str(p)][str(d)] = (pickup_speeds[str(p)]+dropoff_speeds[str(d)])/2

def rate_of_lane(pickup, dropoff):
    return clusters_speeds[str(pickup)][str(dropoff)]


rate_of_lane_ufunc = np.frompyfunc(rate_of_lane, 2, 1)

train.loc[:, 'rate_of_lane'] = rate_of_lane_ufunc(train.pickup_cluster.values,
                                                  train.dropoff_cluster.values)
test.loc[:,'rate_of_lane'] = rate_of_lane_ufunc(test.pickup_cluster.values,
                                                  test.dropoff_cluster.values)

weather_hour = pd.read_csv('Weather.csv')

weather_hour['Datetime'] = pd.to_datetime(weather_hour['pickup_datetime'],
                                          dayfirst=True)
weather_hour['pickup_date'] = weather_hour.Datetime.dt.date
weather_hour['hour'] = weather_hour['Datetime'].dt.hour
weather_hour['fog'] = weather_hour.fog.astype(np.int8)

weather_hour.drop(['pickup_datetime', 'Datetime'], inplace=True, axis=1)
weather_hour = weather_hour.drop_duplicates(subset=['pickup_date', 'hour'],
                                            keep='first')

# maybe turn back to rain and snow will be better?
weather_new = weather_hour[[
    'pickup_date', 'hour', 'pressurem','tempm','hum'
]]

weather_new = weather_new.fillna(method='ffill')  # 临近填充

train = pd.merge(train, weather_new, how='left', on=['pickup_date', 'hour'])
train = train.fillna(method='ffill')
test = pd.merge(test, weather_new, how='left', on=['pickup_date', 'hour'])
test = test.fillna(method='ffill')

fr1 = pd.read_csv(
    'fastest_routes_train_part_1.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
fr2 = pd.read_csv(
    'fastest_routes_train_part_2.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv(
    'fastest_routes_test.csv',
    usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')

def bearing_array(lat1, lng1, lat2, lng2):
    #AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(
        lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values,
                                          train['pickup_longitude'].values,
                                          train['dropoff_latitude'].values,
                                          train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values,
                                         test['pickup_longitude'].values,
                                         test['dropoff_latitude'].values,
                                         test['dropoff_longitude'].values)

train = train.drop([
    'pickup_cluster',
    'dropoff_cluster', 'pickup_datetime',
    'dropoff_datetime', 
    'trip_duration', 'pickup_date',
    'pickup_datetime_group',
    'avg_speed_h','hour'
],
                   axis=1)

test = test.drop([
    'pickup_cluster',
    'dropoff_cluster', 'pickup_datetime',
    'pickup_date','pickup_datetime_group','hour'
],
                 axis=1)

train.to_csv('train_6.csv',index=False)
test.to_csv('test_6.csv',index=False)
