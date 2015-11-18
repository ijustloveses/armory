# encoding: utf-8
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit as SSS

"""
    一些协助预处理数据的函数
    下面函数中的 df 都是 pandas 的 dataframe 变量
"""


def get_numeric_fields(df):
    return df._get_numeric_data().columns


def get_categorical_fields(df, tag_field=None):
    """
        tag_field is always categorical, so ignore it
    """
    numerics = get_numeric_fields(df)
    # "-" is deprecated, use difference() instead
    categoricals = df.columns.difference(numerics)
    if tag_field:
        categoricals = categoricals.difference([tag_field])
    return categoricals


def flatten_categorial_fields(df, cate_fields, other_fields=None, dummy_na=True):
    """
    Example:
    df
            f1 f2
        0   1  T
        1   2  T
        2   3  S
    pd.get_dummies(df, 'f2', dummy_na=True)
            f1  f2_S  f2_T  f2_nan
        0   1     0     1       0
        1   2     0     1       0
        2   3     1     0       0
    pd.get_dummies(df, 'f2')
            f1  f2_S  f2_T
        0   1     0     1
        1   2     0     1
        2   3     1     0

    other_fields 是可能被误认为数字类型的字段，比如年月日，也按 categorical 处理
    """
    # “+” is deprecated, use '|' or .union() instead
    fields = cate_fields | other_fields if other_fields else cate_fields
    return pd.get_dummies(df, fields, dummy_na=dummy_na)


def split_date_fields(df, dfield, yfield='Year', mfield='Month', wfield='Weekday'):
    """
        把日期字段都拆分成年字段、月字段和周字段
        不拆分为日字段，统计意义比较弱
        要求 dfield 是 YYYY-MM-DD 的格式
    """
    # 其实，df[dfield] 本身已经是 pd.Series 类型了，其实貌似不必加转换
    df[dfield] = pd.to_datetime(pd.Series(df[dfield]))
    df[yfield] = df[dfield].apply(lambda x: int(str(x)[:4]))
    df[mfield] = df[dfield].apply(lambda x: int(str(x)[5:7]))
    df[wfield] = df[dfield].dt.dayofweek
    # 删掉原来的日期字段
    df = df.drop(dfield, axis=1)
    return df


def transform_labels(df, other_fields=[]):
    """
        同为处理 categorical features 的方式，和 flatten_categorial_fields 略不同
        后者把 feature 的可选值单拎出来作为新 feature，feature 值为是否等于该值，0/1
        而本函数只是把不同的可选值按顺序修改为 0，1，2，....
    """
    from sklearn import preprocessing
    for f in df.columns:
        if df[f].dtype == 'object' or f in other_fields:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))


def train_test_split(y, test_size=0.1, random_state=1234, n_iter=1):
    splits = SSS(y, test_size=test_size, random_state=random_state, n_iter=n_iter)
    for train_index, test_index in splits:
        return train_index, test_index
