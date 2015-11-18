# encoding: utf-8
"""
自己实现的 grid search + cv
其实还可以参考 xgboost/demo/guide-python/sklearn_examples.py，直接使用 GridSearchCV
"""
import itertools
import xgboost as xgb
import numpy as np
import sklearn.cross_validation as cv
from sklearn.preprocessing import LabelEncoder


class XGBoostGridModel(object):
    def __init__(self, X, labels, param):
        self.X = X
        self.labels = labels
        self.param = param
        self.param['silent'] = param.get('silent', 1)
        classes = list(np.unique(labels))
        num_class = len(classes)
        if num_class > 2:
            self.param['num_class'] = num_class
        classes.sort()
        # xgboost 要求 labels 是 0,1,2 ...
        # Xgboost.XGBClassifier 内置了 LabelEncoder，而 XGBModel 没有内置
        if classes != range(num_class):
            self.le = LabelEncoder()
            self.le.fit(labels)
        else:
            self.le = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_clf_ = None

    def estimate_with_param(self, param, folds):
        """
            对给定的一组参数做 cv，然后求其平均loss
        """
        print "------------- cv fold for params: ", param
        losses = []
        kf = cv.KFold(self.labels.size, n_folds=folds)
        for train_indices, test_indices in kf:
            X_train, X_test = self.X.values[train_indices], self.X.values[test_indices]
            y_train, y_test = self.labels[train_indices], self.labels[test_indices]
            clf = xgb.XGBModel(**param)
            # xgboost 要求 labels 是 0,1,2 ...
            y_train = self.le.transform(y_train) if self.le else y_train
            y_test = self.le.transform(y_test) if self.le else y_test
            clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    eval_metric='logloss', verbose=False)
            evals_result = clf.evals_result()
            # evals_result is like {'validation_0': {'logloss': ['0.68558', '0.67917']}}
            print "    evals_result: ", evals_result
            for e_name, e_mtrs in evals_result.items():
                for e_mtr_name, e_mtr_vals in e_mtrs.items():
                    for loss in e_mtr_vals:
                        losses.append(float(loss))
        avg_loss = sum(losses) / len(losses)
        print "Final average loss is ", avg_loss
        return avg_loss, clf

    def predict_with_param(self, param):
        print "------------- precict with pararms: ", param
        class_probs = self.best_clf_.predict(self.X.values)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self.le.inverse_transform(column_indexes) if self.le else column_indexes

    def search(self, n_folds=5):
        """
            从可选参数中排列组合得到候选的 params 列表
            依次循环列表中的每个 param，做 cv 求 loss
            找到最好的 param
        """
        keys = []
        values = []
        param = {}
        for k, v in self.param.items():
            if isinstance(v, list):
                keys.append(k)
                values.append(v)
            else:
                param[k] = v

        print "self.params: ", self.param
        # 对可选的参数做笛卡尔乘积
        for vals in list(itertools.product(*values)):
            options = {}
            for i, k in enumerate(keys):
                options[k] = vals[i]
            param.update(options)
            print "options :", options
            print "params: ", param
            loss, clf = self.estimate_with_param(param, n_folds)
            print "loss: ", loss, " for options: ", options
            if self.best_score_ is None or self.best_score_ > loss:
                print ">>>> Better options is found"
                self.best_score_ = loss
                self.best_params_ = param
                self.best_clf_ = clf


if __name__ == '__main__':
    import preproc
    import pandas as pd
    train = pd.read_csv('./partial.csv')
    y = train.QuoteConversion_Flag
    train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    train = preproc.split_date_fields(train, 'Original_Quote_Date')
    preproc.transform_labels(train)
    train = train.fillna(-1)

    # XGBModel 的 param 见 __init__() of class XGBModel in the page below:
    # https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py
    param = {'nthread': -1, 'learning_rate': [0.025, 0.25],
             'max_depth': [4, 6], 'objective': 'binary:logistic',
            }
    xgm = XGBoostGridModel(train, y, param)
    xgm.search()
    print "-------------- Final Result ----------------"
    print "best score: ", xgm.best_score_
    print "best params: ", xgm.best_params_

    # 使用最好的结果来做预测
    y_out = xgm.predict_with_param(xgm.best_params_)
    correct = len(filter(lambda x: x, y == y_out))
    print "accuracy: ", round(correct / float(len(y)), 3)
