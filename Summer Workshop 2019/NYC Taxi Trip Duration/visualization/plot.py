import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/Error")

train = pd.read_csv('train.csv')
sub = pd.read_csv('sub.csv')

new = pd.merge(sub,train,how='left',on='id')

new['pickup_datetime'] = pd.to_datetime(new.pickup_datetime)
new.loc[:, 'pickup_date'] = new['pickup_datetime'].dt.date
new['dropoff_datetime'] = pd.to_datetime(new.dropoff_datetime)

# extracting Month, dayofWeek and hour
new['Month'] = new['pickup_datetime'].dt.month

new['hour'] = new['pickup_datetime'].dt.hour

new['dayofweek'] = new['pickup_datetime'].dt.dayofweek

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(
        lat * 0.5)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5)**2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

#avg_speed hour of day, day of week, month of the year

new.loc[:, 'distance_haversine'] = haversine_array(
    new['pickup_latitude'].values, new['pickup_longitude'].values,
    new['dropoff_latitude'].values, new['dropoff_longitude'].values)
new.loc[:, 'avg_speed_h'] = 1000 * new['distance_haversine'] / new['trip_duration']
new.loc[:, 'pred_avg_speed_h'] = 1000 * new['distance_haversine'] / new['pred_trip_duration']

fig, ax = plt.subplots(ncols=3, sharey=True)
ax[0].plot(new.groupby('hour').mean()['pred_avg_speed_h'], 'bo-', lw=2, alpha=0.7, label='pred')
ax[0].plot(new.groupby('hour').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7, label='real')
ax[1].plot(new.groupby('dayofweek').mean()['pred_avg_speed_h'], 'bo-', lw=2, alpha=0.7, label='pred')
ax[1].plot(new.groupby('dayofweek').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7, label='real')
ax[2].plot(new.groupby('Month').mean()['pred_avg_speed_h'], 'bo-', lw=2, alpha=0.7, label='pred')
ax[2].plot(new.groupby('Month').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7, label='real')
ax[0].set_xlabel('Hour of Day')
ax[1].set_xlabel('Day of Week')
ax[2].set_xlabel('Month of Year')
ax[0].set_ylabel('Average Speed')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
       ncol=3, mode="expand", borderaxespad=0.)
fig.suptitle('Average Traffic Speed by Date-part')
plt.show()

