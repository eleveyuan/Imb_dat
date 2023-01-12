import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


class UssSampler:
    def __init__(self, x, y, neighbor=10, plabel=1):
        self.x = np.array(x)
        self.y = np.array(y).ravel()
        self.plabel = plabel  # 少数类标签
        self.neighbor = neighbor

        self.len_pos = len(self.y[self.y == 1])
        self.len_neg = len(self.y[self.y == 0])

        self.eta = self.len_pos / self.len_neg


    def fit_hyperplane(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        # lr = LogisticRegression(max_iter=500)
        lr = GaussianNB()
        lr.fit(self.x, self.y)
        y_pred = lr.predict(self.x)
        print(confusion_matrix(self.y, y_pred))
        print(accuracy_score(self.y, y_pred), '\n', recall_score(self.y, y_pred), '\n', precision_score(self.y, y_pred), '\n')
        prob_matrix = lr.predict_proba(self.x)
        lens = len(prob_matrix)
        M = []
        for i in range(lens):
            if self.y[i] == 0:
                # print(i, prob_matrix[i][0], prob_matrix[i][1])
                pos_gamma = 1 - prob_matrix[i][1]
                neg_gamma = 1 - prob_matrix[i][0]
                neg_gamma_prime = (1- prob_matrix[i][0]) * self.eta
                sensitivity = neg_gamma - (neg_gamma_prime / (pos_gamma + neg_gamma_prime))
                weight = (sensitivity + 1)
                M.append((i, weight))
        M = sorted(M, key=lambda x: -x[1])
        # print(M)
        return [M[i][0] for i in range(len(M))]

    def fit_neighbor(self):
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=self.neighbor)
        knn.fit(self.x)
        nn_matrix = knn.kneighbors(self.x, return_distance=False)
        lens = len(nn_matrix)
        M = []
        for i in range(lens):
            if self.y[i] == 0:
                sensitivity = self.calc_sens(nn_matrix[i])
                weight = (sensitivity + 1)  # + 0.0005 * (prob_matrix[i][0] * prob_matrix[i][1])  # 选择权重，平滑化
                M.append((i, weight))  # 候选集

        M = sorted(M, key=lambda x: -x[1])
        # return [M[i][0] for i in range(len(M)) if i < self.len_pos * 1.2]
        return [M[i][0] for i in range(len(M))]

    def inrobust_idx(self):
        from sklearn.naive_bayes import GaussianNB
        lr1 = GaussianNB()
        lr1.fit(self.x, self.y)
        lr1 = GaussianNB()
        lr1.fit(self.x, self.y)
        y_pred1 = lr1.predict(self.x)

        inro1 = np.argwhere(self.y != y_pred1).ravel()

        from sklearn.linear_model import LogisticRegression
        lr2 = LogisticRegression(max_iter=500)
        lr2.fit(self.x, self.y)
        lr2 = GaussianNB()
        lr2.fit(self.x, self.y)
        y_pred2 = lr2.predict(self.x)

        inro2 = np.argwhere(self.y != y_pred2).ravel()
        return list(np.intersect1d(inro1, inro2))

    def calc_sens(self, row_idx):
        # 通过knn结果模拟对应类别的后验概率
        cnts = len(row_idx)
        pos_cnts = 0
        neg_cnts = 0
        for i, idx in enumerate(row_idx):
            if i == 0:
                continue
            if self.y[idx] == 1:
                pos_cnts += 1
            else:
                neg_cnts += 1
        pos_gamma = pos_cnts / cnts
        neg_gamma = neg_cnts / cnts
        # print(pos_gamma, neg_gamma)
        neg_gamma_prime = self.eta * neg_gamma
        # 敏感度
        sensitivity = (neg_gamma / (pos_gamma + neg_gamma)) - (neg_gamma_prime / (pos_gamma + neg_gamma_prime))
        return sensitivity

    def get_balance_data(self, ratio=1.2):
        minor_idx = None
        if self.is_noise_filter:
            minor_idx = self.filter_noise()
        else:
            minor_idx = list(np.argwhere(self.y == 1).ravel())
        # print(len(minor_idx))
        balan_major_idx = self.fit_neighbor()
        # balan_major_idx = self.fit_hyperplane()
        balan_major_idx = [balan_major_idx[i] for i in range(len(balan_major_idx)) if i < len(minor_idx) * ratio]
        balan_major_idx.extend(minor_idx)

        return self.x[balan_major_idx], self.y[balan_major_idx]

