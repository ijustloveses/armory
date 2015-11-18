# encoding: utf-8

from sklearn import cross_validation
from sklearn import metrics
from collections import defaultdict as dd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
import numpy as np


def run_cv(X, y, clfs, feature_names, n_folds):
    """
        遍历传入的算法 clfs，对每个算法进行 kfold
        每轮 fold 计算一次正确率和 f1，并最后计算平均分
        并展示重要的字段排名
        注：都采用 predict，而不是 predict_proba
    """
    n_samples = X.shape[0]
    kf = cross_validation.KFold(n_samples, n_folds)
    for name, clf in clfs:
        print "\n************* Running ", name, " with ", n_folds, " folds ******************"
        fold = 1
        global_accuracy = 0
        global_f1_score = 0
        for train_index, test_index in kf:
            total = len(test_index)
            print "---------- fold #", fold, ' begins ----------'
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if name == 'Xgboost':
                clf.fit(X_train, y_train, eval_metric="auc")
            else:
                clf.fit(X_train, y_train)
            predict = clf.predict(X_test)

            # mean_accuracy = clf.score(X_test, y_test)
            # print "mean_accuracy: ", mean_accuracy
            # y_test == predict 是个 True & False 的数组 (pd.Series)，filter 其中的 True
            correct1 = len(filter(lambda x: x, y_test == predict))
            accuracy = round(correct1 * 100.0 / total, 3)
            print "accuracy: ", accuracy
            f1score = metrics.f1_score(y_test, predict)
            global_f1_score += f1score
            global_accuracy += accuracy
            fold += 1

        print "\n\t Feature Importances:"    # the features that drive the accuracy
        try:
            for fname, score in sorted(zip(feature_names, clf.feature_importances_),
                                       key=lambda x: x[1], reverse=True):
                if score > 0:
                    print "\t\t" + fname + " :", round(score, 6)
        except:
            print "\t\t n/a"

        avg_accuracy = round(global_accuracy * 1.0 / n_folds, 3)
        global_f1_score = round(global_f1_score / (n_folds + 0.001), 3)
        print "\n======> Average accuracy: "+ str(avg_accuracy) + "%, f1score: ", global_f1_score


def run_blending(X, y, X_submission, clfs, n_folds):
    """
        这是一个两段算法：
        对每个算法做循环：
            做 n_folds cv，n 次训练：
                每次训练后 fit 得到的模型 predict 剩余一份测试集 X*1/n，并 predict X_submission
            n 次训练之后，整个训练样本 X 被完整 predict 一次，而待预测样本集 X_submission 被预测 n 次
            待预测样本 X_submission 取平均值
        循环过后，每个算法得到分别一个 X 和 X_submission 的预测结果，作为新的 X_prime 和 X_submission_prime 的一个分量
        使用 X_prime 和 y 做一个 LR，使用得到的 LR 模型对 X_submission_prime 做预测，得到最终的预测结果

        该函数只用于 binary classificaiton，优先使用 predict_proba，失败则转为使用 predict

        问题在于，nfold cv 中，并没有使用到 y[test_index]，也就是说只使用 cv 建立模型，并没有比对该模型的优劣
        其实，可以先比对 y[test_index]，把结果作为 weight，在第二阶段的 LR 中，使用这个 weight 来调节不同算法的权重
    """
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))    # X_prime 初始化，看到每个算法为每个样本生成一个分量
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))    # X_submission_prime 初始化

    skf = list(StratifiedKFold(y, n_folds))
    for j, clf in enumerate(clfs):
        # 每次 cv fold，会对 X_submission 生成一个预测
        dataset_blend_test_j = np.zeroes((X_submission.shape[0], len(skf)))
        for i, (train_index, test_index) in enumerate(skf):
            X_train, X_test, y_train = X[train_index], X[test_index], y[train_index]
            clf.fit(X_train, y_train)
            try:
                # 第二个分量，也就是标签 1，故此，其实只用于二元分类
                y_submission = clf.predict_proba(X_test)[: 1]
            except:
                y_submission = clf.predict(X_test)
            # 每次 nfold，预测 1/n 的 X
            dataset_blend_train[test_index, j] = y_submission
            # 每次 nfold，都完整预测一次 X_submission
            try:
                dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
            except:
                dataset_blend_test_j[:, i] = clf.predict(X_submission)
        # 对 dataset_blend_test_j 取平均
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    try:
        y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    except:
        y_submission = clf.predict(dataset_blend_test)
    # Linear stretch of predictions to [0,1]
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    return y_submission


def run_bagging(data):
    """
        data 是一组预测结果集，可能取自于不同的算法
        本函数对每个 case，从各个结果集中收集结果标签，取最多的为最终结果
        结果集中的结果都是标签值，而不是概率值
        Drawback : Averaging predictions often reduces overfit
    """
    scores = dd(list)
    for dataset in data:
        for i, result in enumerate(list(dataset)):
            scores[i].append(result)
    # sorted(scores) 返回排好序的 scores.keys()，不返回值; 和对列表的操作结果不同
    final_result = []
    for i in sorted(scores):
        # Counter(xxx).most_common returns something like [('D', 2)]
        final_result.append(Counter(scores[i]).most_common(1)[0][0])
    return final_result


def run_calibrated_ranking(data):
    """
        只用于 binary classification，每个结果集中保存的是概率值
        注意，这个概率值只是用于排序决定位次，而并没有真正用于计算 rank
        具体的 calibration 过程，其实包括了两次的排位位次代替得分，详见代码
        Drawback : Averaging predictions often reduces overfit
        另外，由于排位位次的原因，一半左右的样本最终得分将超过 0.5，只适用均衡分类
    """
    final_result = []
    nsets = len(data)   # data 中 dataset 的个数
    nsamples = len(data[0])  # 每个 dataset 中样本的个数
    all_ranks = dd(list)
    for dataset in data:
        ranks = []
        for i, result in enumerate(list(dataset)):
            # ranks 记录该样本的概率和在结果集中的位置 i
            ranks.append((float(result), i))
        # 对同一结果集中的概率排序，概率小的 -> 排位在前 -> 排名也小
        for rank, item in enumerate(sorted(ranks)):
            # all_ranks 以样本在结果集中的位置为 key，
            # 而 value 中为各个结果集中排位后的排名，看到之后弃用了概率值
            all_ranks[item[1]].append(rank)

    # calibration 过程
    averate_ranks = []
    # 按样本在结果集中的位置为序，
    # 在 averate_ranks 中保存每个样本在各结果集中的平均排名，同时保存位置
    for k in sorted(all_ranks):
        averate_ranks.append((sum(all_ranks[k]) / nsets, k))

    ranked_ranks = []
    # 按平均排名为序来排位，再一次放弃平均排名的结果，而使用排位的位次来作为结果
    # 位次为 0 ~ nsamples-1，故此分母为 nsamples-1，得分将"均匀"分布于 [0~1]
    for rank, k in enumerate(sorted(averate_ranks)):
        ranked_ranks.append((k[1], rank / (nsamples - 1)))
    # 最后，按位置排序，依次记录 calibrationg 后的最终得分
    # 由于是位次，故此接近半数超过 0.5，半数小于 0.5，故此只适用于均衡分类
    for k in sorted(ranked_ranks):
        final_result.append(ranked_ranks[k])
    return final_result
