#!/usr/bin/env python
# encoding: utf-8

from sklearn.feature_extraction.text import CountVectorizer   # TfidfVectorizer

tv = CountVectorizer(encoding=u'utf-8', analyzer=u'char_wb')
x = [u'你拍一我拍一', u'一个小孩']
r = tv.fit_transform(x)

# 查看词典，结果如下：
"""
一
1
---------

0
---------
孩
4
---------
个
2
---------
拍
7
---------
小
5
---------
我
6
---------
你
3
"""
for k, v in tv.vocabulary_.iteritems():
    print k
    print v
    print '---------'

# 类似的，也可以这样按先后顺序看词典
"""

一
个
你
孩
小
我
拍
"""
for k in tv.get_feature_names():
    print k

# 得到的是一个 scipy sparse matrix
"""
   <2x8 sparse matrix of type '<type 'numpy.float64'>'
   with 10 stored elements in Compressed Sparse Row format>
"""
print r.__class__

# 这样查看结果是不正确的姿势
""" array([1, 2, 2, 1, 2, 1, 1, 1, 1, 2]) """
print r.data

# 需要调整为 dense 矩阵，才是正确的
"""
matrix([[2, 2, 0, 1, 0, 0, 1, 2],
        [2, 1, 1, 0, 1, 1, 0, 0]])
"""
print r.todense()

# 我们可以比较一下 dense 、词典 和原语料
"""
语料: [u'你拍一我拍一', u'一个小孩']
词典 + dense 结果的矩阵

词典 index   0     1    2    3    4    5    6    7
词典词       ''    一   个   你   孩   小   我   拍
dense 结果   2     2    0    1    0    0    1    2    ==> 你拍一我拍一 加上两个 ''
             2     1    1    0    1    1    0    0    ==> 一个小孩 加上两个 ''
"""

# 尝试一下 transform
w = tv.transform([u'两个小孩拍'])
print w.todense()
"""
matrix([[2, 0, 1, 0, 1, 1, 0, 1]])
对比上面的词典，正好是 "个小孩" 加上两个 ''，注意由于 "两" 不在词典中，故此被忽略
"""
