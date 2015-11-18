# encoding: utf-8
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression
import pandas as pd
from armory import preproc, proc


train = pd.read_csv("partial.csv")

cate_fields = preproc.get_categorical_fields(train, tag_field="QuoteConversion_Flag")
train = preproc.flatten_categorial_fields(train, cate_fields)
train = train.fillna(0)

feature_names = train.columns
y = train.QuoteConversion_Flag
X = train.drop("QuoteConversion_Flag", axis=1)
X = X.values
n_estimators = 50
n_folds = 2

clfs = [
        ['LR', LogisticRegression(penalty='l1', random_state=123)],
        ['Ada', AdaBoostClassifier(n_estimators=n_estimators,random_state=123)],
        ['GBC', GradientBoostingClassifier(n_estimators=n_estimators,random_state=123)],
        ['ET', ExtraTreesClassifier(n_estimators=n_estimators,random_state=123)],
        ['RFC', RandomForestClassifier(n_estimators=n_estimators, random_state=123)],
       ]
proc.run_cv(X, y, clfs, feature_names, n_folds)
