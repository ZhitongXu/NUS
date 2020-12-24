import os
import pandas as pd
import numpy as np
import lightgbm as lgb
os.chdir("C:/Users/Rhodia/Downloads/workshop/NYC_Taxi/data")

lgb_model = lgb.Booster(model_file='lightgbm_temp.model')

test = pd.read_csv('test_features_modify.csv')
X_test = test.drop(['id'],axis=1)
test_id = test['id']

predictions = lgb_model.predict(X_test)


#Create a data frame designed a submission on Kaggle
submission = pd.DataFrame({'id': test_id, 'trip_duration': np.exp(predictions)-1})
submission.head()

#Create a csv out of the submission data frame
submission.to_csv("sub.csv", index=False)