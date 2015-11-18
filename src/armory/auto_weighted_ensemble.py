# encoding: utf-8
"""
    自动学习多个算法 ensemble 的权重
    使用的子算法需要实现 predict_proba 方法
"""
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import preproc


class AutoWeightedEnsemble(object):
    def __init__(self, clfs):
        self.clfs = clfs

    def learn_weight(self, train, labels, method='SLSQP'):
        train_index, test_index = preproc.train_test_split(labels)
        train_x, train_y = train.values[train_index], labels[train_index]
        test_x, test_y = train.values[test_index], labels[test_index]

        # run clfs one by one
        predictions = []
        for name, clf in self.clfs:
            clf.fit(train_x, train_y)
            prediction = clf.predict_proba(test_x)
            predictions.append(prediction)
            print '{name}: logloss score is {score}'.format(name=name, score=log_loss(test_y, prediction))

        # learn weight by minimize loss
        def log_loss_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = 0
            for weight, prediction in zip(weights, predictions):
                final_prediction += weight * prediction
                return log_loss(test_y, final_prediction)

        # In practice, we should choose many random starting weights and run minimize a few times
        starting_values = [1 / float(len(predictions))] * len(predictions)
        # adding constraints
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        # bound weight between 0 and 1
        bounds = [(0, 1)] * len(predictions)

        # 做优化，更新 weight
        res = minimize(log_loss_func, starting_values, method=method, bounds=bounds, constraints=cons)
        print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
        print('Best weights: {weights}'.format(weights=res['x']))
        print res


if __name__ == '__main__':
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.linear_model import LogisticRegression as LR

    train = pd.read_csv('./partial.csv')
    y = train.QuoteConversion_Flag
    train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    train = preproc.split_date_fields(train, 'Original_Quote_Date')
    preproc.transform_labels(train)
    train = train.fillna(-1)

    rfc = RFC(n_estimators=50, random_state=4141, n_jobs=-1)
    logreg = LR()
    rfc2 = RFC(n_estimators=50, random_state=1337, n_jobs=-1)

    awe = AutoWeightedEnsemble([['rfc', rfc], ['logreg', logreg], ['rfc2', rfc2]])
    awe.learn_weight(train, y)
