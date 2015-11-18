# encoding: utf-8

import pandas as pd


def save_submission_from_file(oldfile, submissionfile, preds, tag):
    """
        oldfile:        已有提交文件，首行为字段名，其中有一列为结果标签列
        submissionfile: 待生成提交文件，将和 oldfile 保持同一格式
        preds:          预测结果列表
        tag:            标签字段
    """
    sample = pd.read_csv(oldfile)
    sample[tag] = preds
    sample.to_csv(submissionfile, index=False)


def save_submission_from_data(submissionfile, headers, data, delimiter=','):
    """
        submissionfile: 待生成提交文件
        headers:        header line 文字
        data:           各列数据列表的列表
        delimiter:      分隔符
    """
    assert len(headers) == len(data)
    with open(submissionfile) as f:
        f.write(delimiter.join(headers) + '\n')
        for i, no in enumerate(list(data[0])):
            f.write(delimiter.join([str(d[i]) for d in data]) + '\n')
