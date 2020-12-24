import os
import pandas as pd
import numpy as np
import joblib
os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")
train = pd.read_csv('train_features.csv')
test = pd.read_csv('test_features.csv')

X_train = train.drop(['id','log_trip_duration'],axis=1)
y_train = train['log_trip_duration']

X_test = test.drop(['id'],axis=1)
test_id = test['id']

#lgb_model = joblib.load("train_model.m")

import lightgbm as lgb

lgb_params = {
    'boosting_type': 'gbdt', 
    'objective': 'regression', 

    'learning_rate': 0.02, 
    'num_leaves': 49, 
    'max_depth': 7,
    'min_data_in_leaf': 16,
    'n_estimators':791,
    'n_estimators': 829, 
    'feature_fraction': 0.5,
    'bagging_fraction': 1, 
    'max_bin': 1000
}


#Training on all labeled data using the best parameters
lgb_df = lgb.Dataset(X_train, y_train)
lgb_model = lgb.train(lgb_params, lgb_df, num_boost_round=1500)



predictions = lgb_model.predict(X_test)

#Create a data frame designed a submission on Kaggle
submission = pd.DataFrame({'id': test_id, 'trip_duration': np.exp(predictions)-1})
submission.head()

#Create a csv out of the submission data frame
submission.to_csv("sub.csv", index=False)

#joblib.dump(lgb_model,"lgbm_train_model.m")