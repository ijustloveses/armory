#!/usr/bin/env python
# encoding: utf-8

import os
import shutil
import gensim
from glob import glob
import numpy as np
import pandas as pd
from copy import deepcopy


class Feeder(object):
    """
    要求 Feeder 可以把数据拆分为训练和测试数据
    types 变量包含所有分类标签
    提供 feed 方法，可以指定标签返回数据，也可以返回全数据
    对于 train 类型，返回句子流；对于 test 类型，按样本返回句子列表、分类、样本id 等信息
    """
    def __init__(self):
        self.types = []
        self.train_data = [['name1', 'label1', ['x', 'y'], [['a', 'b'], ['c', 'd']]],
                           ['name2', 'label2', ['x', 'y'], [['a', 'b'], ['c', 'd']]]]
        self.test_data = [['name1', 'label1', ['x', 'y'], [['a', 'b'], ['c', 'd']]],
                          ['name2', 'label2', ['x', 'y'], [['a', 'b'], ['c', 'd']]]]
        self.types = list(set([r[1] for r in self.train_data]))

    def feed(self, type=None, output_type="sentence", datatype="train"):
        data = self.train_data if datatype == "train" else self.test_data
        for fname, label, title, content in data:
            if output_type == "sentence":    # train
                yield title
                for c in content:
                    yield c
            elif output_type == "sentencelist":    # test
                sentencelist = content + [title]
                yield {'list': sentencelist, 'type': label, 'fname': fname}


class W2VClassifier(object):
    def __init__(self, feeder, rootdir, dirname='attempt.1'):
        self.rs = feeder
        self.models = {}
        self.root = os.path.join(rootdir, dirname)

    def prepare_dir(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.mkdir(self.root)

    def build_vocab(self):
        self.base_model = gensim.models.Word2Vec(size=100, window=7, min_count=5, workers=8, hs=1, negative=0)
        print "building vocabulary ..."
        self.base_model.build_vocab(self.rs.feed())

    def train_word2vec_models(self):
        for t in self.rs.types:
            print u"training type: {} ...".format(t)
            model = deepcopy(self.base_model)
            model.train(self.rs.feed(type=t))
            model.save(os.path.join(self.root, u"{}.model".format(t)))
            self.models[t] = model

    def predict_test_set(self):
        print "segmenting test set ..."
        y = {}
        sentencelist = []
        doc_index = []
        for rpt in self.rs.feed(output_type="sentencelist", datatype="test"):
            y[rpt['fname']] = rpt['type']
            sentencelist += rpt['list']
            doc_index += [rpt['fname']] * len(rpt['list'])

        types = [t for t, _ in self.models.iteritems()]
        print "scoring test set ..."
        llhd = np.array([m.score(sentencelist, len(sentencelist)) for _, m in self.models.iteritems()])
        print "calculating probs ..."
        lhd = np.exp(llhd - llhd.max(axis=0))
        prob = pd.DataFrame((lhd / lhd.sum(axis=0)).transpose(), columns=types)
        prob['doc'] = doc_index
        self.prob = prob.groupby('doc').mean()   # doc will be added and set to index
        rpts = self.prob.index.tolist()
        self.y = [y[r] for r in rpts]

    def show_result(self):       # 结果可以使用 categorical_evaluator.py 来更好的估计
        max_prob_type = self.prob.idxmax(axis=1)
        result = max_prob_type == self.y
        corrected = result[result == True].count()
        total = len(result)
        print "accuracy overall: {} -- {} out of {}".format(corrected / float(total), corrected, total)

    def run(self):
        self.prepare_dir()
        self.build_vocab()
        self.train_word2vec_models()
        self.predict_test_set()
        self.show_result()

    def load_models(self):
        modfiles = glob(os.path.join(self.root, u"*.model"))
        self.models = {}
        for mf in modfiles:
            # mf like u'/storage/w2v/analyst_report/models/attempt.1/\u4e16\u754c\u7ecf\u6d4e\u7814\u7a76\u62a5\u544a.model'
            t = os.path.basename(mf)[:-6]
            self.models[t] = gensim.models.Word2Vec.load(mf)
