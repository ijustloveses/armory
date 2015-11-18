# encoding: utf-8

import pandas as pd
import xgboost as xgb
import preproc
import proc


train = pd.read_csv('./partial.csv')
y = train.QuoteConversion_Flag
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
# xgb expect float features, so convert all non-float features
train = preproc.split_date_fields(train, 'Original_Quote_Date')
preproc.transform_labels(train)
train = train.fillna(-1)

feature_names = train.columns
train = train.values
n_folds = 2
clfs = [['Xgboost', xgb.XGBClassifier(n_estimators=25,
        nthread=-1,
        max_depth=10,
        learning_rate=0.025,
        silent=True,
        subsample=0.8,
        colsample_bytree=0.8)]]
proc.run_cv(train, y, clfs, feature_names, n_folds)
