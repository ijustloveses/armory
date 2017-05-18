#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd


class CategoricalEvaluator(object):
    def __init__(self, probs, y, types):
        """
        probs: pandas dataframe with columns named as categories. axis-0 represents categories, axis-1 represents samples
                      cat1    cat2    cat3    cat4 ...
             sample1   p11     p12     p13     p14 ...
             sample2   p21     p22     p23     p24 ...
             ..................
        y: python list or np.array, samples' real labels
        types: distinct labels
        """
        self.probs = probs
        self.y = y
        self.types = types

    def show_top1_result(self):
        max_prob_type = self.probs.idxmax(axis=1)  # pandas.core.series.Series
        result = max_prob_type == self.y           # pandas.core.series.Series
        corrected = result[result == True].count()
        total = len(result)                        # or result.count()
        print "accuracy overall: {} -- {} out of {}".format(corrected / float(total), corrected, total)

        resdf = result.to_frame(name="result")
        resdf['y'] = self.y
        resdf['y_pred'] = max_prob_type
        for t in self.types:
            print u"----- {} -----".format(t)
            predicted = len(resdf[resdf['y_pred'] == t])
            labeled = len(resdf[resdf['y'] == t])
            corrected = len(resdf[resdf['y_pred'] == t][resdf['result'] == True])
            print u"precision: {} -- {} out of {}".format(0 if predicted == 0 else float(corrected) / predicted, corrected, predicted)
            print u"recall: {} -- {} out of {}".format(0 if labeled == 0 else float(corrected) / labeled, corrected, labeled)

    def show_topn_result(self, n):
        order = np.argsort(-self.probs, axis=1).iloc[:, :n]    # 每行分别排序，取前 n 列；得到一个 numpy.array，每个元素是列号，注意不是列名
        result = pd.DataFrame(self.probs.columns[order],       # 列号 ==> 列名 转换
                              index=self.probs.index)          # 保持 index 不变
        in_topn = [val in result.iloc[i].tolist() for i, val in enumerate(self.y)]
        resdf = pd.DataFrame({'result': in_topn, 'y': self.y})

        corrected = len(resdf[resdf['result'] == True])
        total = len(resdf)
        print "accuracy overall: {} -- {} out of {}".format(corrected / float(total), corrected, total)
        for t in self.types:
            print u"----- {} -----".format(t)
            filtered = resdf[resdf['y'] == t]
            labeled = len(filtered)
            corrected = len(filtered[filtered['result'] == True])
            print u"recall: {} -- {} out of {}".format(0 if labeled == 0 else float(corrected) / labeled, corrected, labeled)
