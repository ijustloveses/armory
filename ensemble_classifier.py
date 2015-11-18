# encoding: utf-8
"""
    Reference:
    http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
"""
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.pipeline import _name_estimators
import numpy as np


class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, clfs, voting='hard', weights=None):
        """
            voting: if 'hard', uses predicted class labels for majority rule voting
                    if 'soft', predicts the class label based on the argmax of the sums of the predicted probalities
        """
        self.clfs = clfs
        # _name_estimators([LogisticRegression()]) ==> [('logisticregression', LogisticRegression(C=1.0,  ....))]
        self.named_clfs = {key:value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        # only for binary classification
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output classification is not supported')
        self.le_ = LabelEncoder()
        self.le_.fit(y)    # 把 y 的标签转为 0/1
        self.classes_ = self.le_.classes_
        self.clfs_ = []
        for clf in self.clfs:
            fitted_clf = clone(clf).fit(X, self.le_.transform(y))
            self.clfs_.append(fitted_clf)
        return self

    def predict(self, X):
        if self.voting == 'soft':
            """
                把得到的 样本 X 标签 2维向量，按 axis=1 也就是第二维度去 max，也就是取最大概率的标签
                得到一维，也就是样本维度的向量，取值为最大概率(max) 对应的那个标签(argmax)
            """
            maj = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = self._predict(X)
            # 首先看 bincount like: np.argmax(np.bincount([1, 2, 2], weights=[3, 1, 1]))
            # 就是创建从 0 开始的桶，输入 [1, 2, 2] 分别对应 1号桶和2号桶，分别为一次和两次
            # 对应次的权重为 3，1，1，那么就是说 1号桶1次，权重3； 2号桶2次，权重1
            # 返回 1，指1号桶的加权次数最大
            # 分解来看，np.bincount([1, 2, 2], weights=[3, 1, 1]) = [0, 3, 2] 对应 0，1，2 号桶的加权次数
            # 故此，最大的 max 为 3，最大值对应的桶 argmax 为 1
            # 其次，看 np.apply_along_axis 沿着 axis=1 也就是算法的维度做上面的 argmax(bincount)
            # 也就是说，每个样本保持维度不变，取每个算法的结果得到类似 [0,1,0,....]，然后再在上面做 weights 的 bincount
            # 最后取加权后最大的标签作为该样本的最终标签结果：样本维度的一维向量
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                      axis=1, arr=predictions)
        maj = self.le_.inverse_transform(maj)    # 再把 0、1、... 转换回到原来的标签
        return maj

    # 预测的结果按 weight 做平均
    def predict_proba(self, X):
        """
        假设有 3 个算法，X 有两个样本 A/B，两类标签 0/1
        那么 self._predict_probas 返回 3 个元素，对应每个算法的结果
        每个元素有两个子元素，每个子元素对应 X 中的一个样本的预测结果
        每个预测结果为 clf.predict_proba 返回的一个列表，分别为对应两个标签的概率
        举例如下：
        >>> weight = [1,1,2]
        >>> pred = np.asarray([[[0.6, 0.4], [0.7, 0.3]], [[0.5, 0.5], [0.6, 0.4]], [[0.4, 0.6], [0.5, 0.5]]])
        其中，[[0.6, 0.4], [0.7, 0.3]] 为第一个算法的结果， [0.6, 0.4] 为样本 A 不同分类的概率，[0.7, 0.3] 则为样本 B
        故此，3 X 2 X 2 维向量，分别对应 算法、样本、标签
        >>> avg = np.average(pred, axis=0, weights=weight)
        按 axis=0 轴做平均，weights=[1,1,2]，实际上也就是按不同算法做平均
        于是，不同算法下，第一个元素取标签 0 的平均几率为: (0.6*1 + 0.5*1 + 0.4*2) / (1+1+2) = 1.9/4 = 0.475
        >>> avg
        array([[ 0.475,  0.525],
           [ 0.575,  0.425]])      得到的是 2 X 2 也就是 样本 X 标签
        """
        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def _predict_probas(self, X):
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])

    def _predict(self, X):
        """ 显然，这个得到的是 算法 X 样本二维的向量，再转置为 样本 X 算法二维向量 """
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def transform(self, X):
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        if not deep:
            return super(EnsembleClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for k, v in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, k)] = v
            return out


if __name__ == '__main__':
    from sklearn import cross_validation
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import preproc
    import proc

    train = pd.read_csv('./partial.csv')
    y = train.QuoteConversion_Flag
    train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    train = preproc.split_date_fields(train, 'Original_Quote_Date')
    preproc.transform_labels(train)
    X = train.fillna(-1)

    np.random.seed(123)
    clf1 = LogisticRegression()
    clf2 = RandomForestClassifier()
    clf3 = GaussianNB()
    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    print "Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), "Emsemble")

    # ensemble of ensemble
    eclf1 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[5, 2, 1])
    eclf2 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[4, 2, 1])
    eclf3 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1, 2, 4])
    eclf = EnsembleClassifier(clfs=[eclf1, eclf2, eclf3], voting='soft', weights=[2, 1, 1])
    scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'Ensemble of ensemble'))

    # grid search
    from sklearn.grid_search import GridSearchCV
    eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    params = {'logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}
    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(X, y)
    for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))

    print "\nProc.cv running..."
    feature_names = X.columns
    n_folds = 2
    proc.run_cv(X.values, y, [['Eclf', eclf]], feature_names, n_folds)
