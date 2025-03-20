#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project: meta-baseline 
@File: __init__.py.py
@Date: 2024/3/8 15:00 
@Author: Lemon
'''
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class MyKMeans(object):
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_features = None
        self.cluster_centers_ = None

    def _get_cosine_similarity(self, X, Y):
        # 计算余弦相似度
        # 类型转换
        X = np.array(X)
        Y = np.array(Y)
        # 参数校验
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
            raise ValueError("输入数据必须二维数组")

        return (X[:, np.newaxis] * Y).sum(axis=-1) / np.linalg.norm(x=Y, axis=-1)[np.newaxis, :] / np.linalg.norm(x=X, axis=-1)[:, np.newaxis]

    def fit(self, X):
        # 训练模型

        # 类型转换
        X = np.array(X)

        # 参数校验
        if X.ndim != 2:
            raise ValueError("输入数据必须二维数组")

        # 记录特征个数
        self.n_features = X.shape[1]

        # 初始化分类中心
        indices = np.random.choice(a=len(X),
                         size = self.n_clusters,
                         replace = False)
        self.cluster_centers_ = X[indices]

        for step in range(self.max_iter):
            # 计算距离
            # distances = np.linalg.norm(x=X[:, np.newaxis] - self.cluster_centers_,
            #                axis=-1)

            # 计算余弦相似度
            cosine_similarities = self._get_cosine_similarity(X=X, Y=self.cluster_centers_)

            # 划分类别
            self.labels = np.argmax(a=cosine_similarities,
                      axis=-1)

            # 重新计算分类中心
            cluster_centers_ = np.array([X[self.labels == idx].mean(axis=0)for idx in range(self.n_clusters)])

            # 判定是否继续迭代
            if np.allclose(a=self.cluster_centers_, b=cluster_centers_):
                break
            else:
                self.cluster_centers_ = cluster_centers_

    def predict(self, X):
        # 推理

        # 类型转换
        X = np.array(X)
        # 参数校验
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError("输入数据必须二维数组，且必须与训练数据相同维度")

        # 计算余弦相似度
        cosine_similarities = self._get_cosine_similarity(X=X, Y=self.cluster_centers_)

        # 划分类别
        labels = np.argmax(a=cosine_similarities,
                           axis=-1)
        return labels


def MyKmeans(feat, proto):
    # 1.feat, proto转换为 NumPy 数组
    feat = feat.cpu().numpy()
    proto = proto.cpu().numpy()

    # 2.遍历feat的episode，对于每个episode
    n_episodes = feat.shape[0]  # 2
    n_acc1 = 0
    for idx in range(n_episodes):
        # 2.1 形状压缩为二维
        idx_feat = feat[idx]  # 这时一集的数据，对它进行聚类，并进行分类。
        idx_proto = proto[idx]

        # 2.2 新建 K-means 对象
        km = MyKMeans(n_clusters=5)

        # 2.3 将查询图片feat，喂进去
        X = np.concatenate((idx_proto, idx_feat), axis=0)
        km.fit(X=X)

        # 2.3 将查询图片feat，喂进去
        # km.fit(X=idx_feat)

        n_acc1 += allocation_by_metric(km, proto, idx) # v2分配(allocation_by_metric):智能分配,已被分配了类原型的簇将不会再得到类原型.
        # n_acc1 += allocation_by_cluster(km, idx_proto) # 聚类分配(allocation_by_cluster)
    return n_acc1


def allocation_by_metric(kmeans, proto, n_way, n_query):
    # 1.计算相互距离,得到距离矩阵all_distances
    ################### 这里进行改进，用不同的方式得到5*5的举例矩阵 ##############################
    # 方式1，原型和聚类中心用余弦距离
    all_distances = cosine_similarity(proto, kmeans.cluster_centers_)

    # 方式2，原型和聚类中心用欧式距离
    # all_distances = np.linalg.norm(centroids[:, np.newaxis, :] - proto[idx], axis=2).T
    #################################################

    # 2.将原型分配给簇, 每个原型仅分配给一个簇, 每个簇最多得到一个原型, 最后记录得到fix_dict字典
    fix_dict = {}  # 用来修改预测标签的字典：聚类标签-->目标标签
    for kk in range(5):  # 执行5次，完成5次分类
        min_index = np.unravel_index(np.argmax(all_distances), all_distances.shape)
        # print('最小值坐在位置',min_index,type(min_index))
        fix_dict[min_index[1]] = min_index[0]
        for kk1 in range(5):  # 修改最小值所在行和列的值为999，避免下次被min选择。
            all_distances[min_index[0]][kk1] = 0  # 做了归一化，值是1~2，没做5~13
            all_distances[kk1][min_index[1]] = 0

    # 3.将聚类标签修改为针织标签
    # 3.1 获取聚类标签
    true_labels = kmeans.labels_.copy()
    # 3.2 用fix_dict字典对聚类的标签进行修改, 得到的就是真实标签
    for i in range(len(true_labels)):
        true_labels[i] = fix_dict[true_labels[i]]

    print('标签预测结果: ', fix_dict)
    print('聚类标签: ', kmeans.labels_)
    print('真实标签: ', true_labels)

    # 开始统计acc
    label = np.array([])
    for i in range(n_way):
        for j in range(n_query):
            label = np.append(label, i)

    # 计算相同元素的个数
    count = np.count_nonzero(label == true_labels)
    print('count=', count)
    return count


def allocation_by_metric_norm(kmeans, proto, n_way, n_query):
    # 1.计算相互距离,得到距离矩阵all_distances
    ################### 这里进行改进，用不同的方式得到5*5的举例矩阵 ##############################
    # 方式1，原型和聚类中心用余弦距离
    all_distances = cosine_similarity(proto, kmeans.cluster_centers_)

    # 方式2，原型和聚类中心用欧式距离
    # all_distances = np.linalg.norm(centroids[:, np.newaxis, :] - proto[idx], axis=2).T
    #################################################

    # 2.将原型分配给簇, 每个原型仅分配给一个簇, 每个簇最多得到一个原型, 最后记录得到fix_dict字典
    fix_dict = {}  # 用来修改预测标签的字典：聚类标签(键)--> 真实标签(值)
    for i in range(5):  # 执行5次，完成5次分类
        min_index = np.unravel_index(np.argmax(all_distances), all_distances.shape)
        fix_dict[min_index[1]] = min_index[0]
        for j in range(5):  # 修改最小值所在行和列的值为999，避免下次被min选择。
            all_distances[min_index[0]][j] = 0  # 做了归一化，值是1~2，没做5~13
            all_distances[j][min_index[1]] = 0

    # 3.将聚类标签修改为针织标签
    # 3.1 获取聚类标签
    true_labels = kmeans.labels_.copy()
    # 3.2 用fix_dict字典对聚类的标签进行修改, 得到的就是真实标签
    for i in range(len(true_labels)):
        true_labels[i] = fix_dict[true_labels[i]]

    # print('标签预测结果: ', fix_dict)
    # print('聚类标签: ', kmeans.labels_)
    # print('真实标签: ', true_labels)

    # 4.根据真实标签设计伪logits分数,如一个样本的标签为3,则其logits分数为[0,0,0,1,0]
    logits = np.zeros((n_way * n_query, n_way))
    # 遍历true_labels, 模拟好logits
    for i in range(len(true_labels)):
        col = true_labels[i]
        logits[i][col] = 1

    return logits


def allocation_by_cluster(kmeans, proto, n_way, n_query):
    answers = kmeans.predict(proto)
    label_answers = np.array([])

    for i in range(n_way):
        for j in range(n_query):
            label_answers = np.append(label_answers, answers[i])

    # 答案比对
    count = np.count_nonzero(kmeans.labels_ == label_answers)
    print('==================')
    print(answers)
    print(kmeans.labels_)
    print(label_answers)
    print('==================')
    print('count=', count)
    return count


def bmm(feat, proto):
    # 0.准备变量
    n_acc1 = 0

    version = 3  # 1是metric,2是cluster,3是metric_norm

    n_way = int(proto.shape[1])
    n_query = int(feat.shape[1] / n_way)

    logits = np.zeros((1, n_way * n_query, n_way))

    # 1.feat, proto转换为 NumPy 数组
    feat = feat.cpu().numpy()
    proto = proto.cpu().numpy()

    # 2.遍历feat的episode，对于每个episode
    n_episodes = feat.shape[0]  # 2
    for idx in range(n_episodes):
        # 2.1 形状压缩为二维
        idx_feat = feat[idx]  # 这时一集的数据，对它进行聚类，并进行分类。
        idx_proto = proto[idx]
        # print(type(idx_feat), idx_feat.shape, idx_feat.ndim)
        # 2.2 新建 K-means 对象
        # kmeans = KMeans(n_clusters=5, n_init=10)
        kmeans = KMeans(n_clusters=5, init=idx_proto, n_init=1)

        # 2.3 将查询图片idx_feat，支持集原型idx_proto，喂进去
        # X = np.concatenate((idx_proto, idx_feat), axis=0)
        # kmeans.fit(X)

        # 2.3 将查询图片feat，喂进去
        kmeans.fit(idx_feat)

        if version == 1:
            n_acc1 += allocation_by_metric(kmeans, idx_proto, n_way, n_query) # v2分配方式
        elif version == 2:
            n_acc1 += allocation_by_cluster(kmeans, idx_proto, n_way, n_query) # 将proto投射到聚类中间，得到聚类，根据聚类结果进行更改
        elif version == 3:
            idx_logits = allocation_by_metric_norm(kmeans, idx_proto, n_way, n_query)  # v2分配+统一度量方式
            idx_logits = idx_logits[np.newaxis, :, :]
            # 拼接logits
            logits = np.concatenate((logits, idx_logits), axis=0)

    if version == 3:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.from_numpy(logits[1:, :, :]).to(device)
        return logits
    return n_acc1
