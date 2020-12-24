import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

pickup = pd.concat((train,test))[['pickup_latitude','pickup_longitude']]
dropoff = pd.concat((train,test))[['dropoff_latitude','dropoff_longitude']]

coords = np.vstack((pickup[['pickup_latitude', 'pickup_longitude']].values,
                    dropoff[['dropoff_latitude', 'dropoff_longitude']].values))

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

df_all = pd.concat((train, test))[['pickup_cluster', 'dropoff_cluster']]

cnt_pickup_dropoff = df_all.groupby(by=['pickup_cluster','dropoff_cluster'])
newdf = cnt_pickup_dropoff.size()
newdf = newdf.reset_index(name='cnt_pickup_dropoff')
train = pd.merge(train,newdf,how='left',on=['pickup_cluster','dropoff_cluster'])

def no_route(x):
    return 'route_' + str(x['pickup_cluster']) + '_' + str(x['dropoff_cluster'])

train['route'] = train.apply(no_route,axis=1)

train.to_csv('his.csv',index=False)
