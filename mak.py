from scipy import sparse
from sklearn.utils import check_X_y

try:
    from sklearn.utils import safe_indexing
except ImportError:
    from sklearn.utils import _safe_indexing as safe_indexing

from validation import check_target_type, check_ratio
import numpy as np


# 求给定矩阵中的样本点到中心点的马氏距离
def mashi_distance(x_array):
    # 给定矩阵的样本中心点
    x_mean = np.mean(x_array, axis=0)
    # 给定矩阵的协方差矩阵
    S = np.cov(x_array.T)
    ma_distances = []
    if np.linalg.det(S) != 0:
        for x_item in x_array:
            SI = np.linalg.inv(S)
            delta = x_item - x_mean
            # 给定矩阵中的相应样本点到中心点的马氏距离
            distance = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            # 这里是项目要求得到马氏距离的平方
            ma_distances.append(distance ** 2)
    else:
        print("矩阵行列式为0")
    return ma_distances


class MAHAKIL:
    def __init__(self, ratio='auto', sampling_type="over-sampling"):
        self.ratio = ratio
        self.sampling_type = sampling_type

    # 产出新样本前对所给数组x_old，y_old进行检测，看其长度，类型是否一致
    def fit(self, x_old, y_old):
        y_old = check_target_type(y_old)

        x_check,  y_check = check_X_y(
            x_old, y_old, accept_sparse=['csr', 'csc'])
        # ratio_xy为少数类要产生新样本的数目
        self.ratio_xy = check_ratio(self.ratio, y_check, self.sampling_type)
        return self

    def sample(self, x_old, y_old):
        X_resampled = x_old.copy()
        y_resampled = y_old.copy()
        for class_sample, n_samples in self.ratio_xy.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y_old == class_sample)
            X_class = safe_indexing(x_old, target_class_indices)
            X_new, y_new = self.make_samples(X_class, class_sample, n_samples)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))
        return X_resampled, y_resampled

    # MAHAKIL方法具体产生新样本的方式
    def make_samples(self, X_class, class_sample, n_samples):
        x_row = np.shape(X_class)[0]
        # 得到X_class数组的相关马氏距离
        mashi_distances = mashi_distance(X_class)
        # 将X_class数组里的每个样本和其马氏距离保存在mashi_zip数组里
        mashi_zip = zip(X_class, mashi_distances)
        # 将mashi_zip数组按其保存的马氏距离从大到小排序
        sample_arr = sorted(mashi_zip, key=lambda x: x[1], reverse=True)
        Nmid = int(x_row / 2)
        nb1 = []
        nb2 = []
        for i in range(Nmid):
            nb1.append(sample_arr[i][0])
        Nbin1 = list(zip(nb1, range(Nmid)))
        for j in range(Nmid, x_row):
            nb2.append(sample_arr[j - Nmid - 1][0])
        Nbin2 = list(zip(nb2, range(Nmid)))
        x_new_list = []
        xre_list = []
        nmid = 0
        for i in range(Nmid):
            x_reshape = (np.array(Nbin1[i][0]) + np.array(Nbin2[i][0])) * 0.5
            xre_list.append(x_reshape)
            nmid += 1
            if (len(x_new_list) + len(xre_list)) >= n_samples:
                break
        x_new_list.extend(list(zip(xre_list, range(nmid))))
        if len(x_new_list) >= n_samples:
            y_new = np.array([class_sample] * len(x_new_list))
            return xre_list, y_new
        x_new_copyl = x_new_list.copy()
        x_new_copyr = x_new_list.copy()
        nmid = 0
        # 将第一代祖先不断与后面的子孙样本点结合产生新样本，知道满足数量n_sampes
        while len(x_new_list) < n_samples:
            xleft_list = []
            xright_list = []
            for i in range(Nmid):
                x_reshape = (np.array(Nbin1[i][0]) +
                             np.array(x_new_copyl[i][0])) * 0.5
                xleft_list.append(x_reshape)
                nmid += 1
                if (len(x_new_list) + len(xleft_list)) >= n_samples:
                    break
            x_new_copyl = list(zip(xleft_list, range(nmid)))
            x_new_list.extend(x_new_copyl)
            if (len(x_new_list) + len(xleft_list)) < n_samples:
                nmid = 0
                for j in range(Nmid):
                    x_reshape = (
                        np.array(Nbin2[j][0]) + np.array(x_new_copyr[j][0])) * 0.5
                    xright_list.append(x_reshape)
                    nmid += 1
                    if (len(x_new_list) + len(xright_list)) >= n_samples:
                        break
                x_new_copyr = list(zip(xleft_list, range(nmid)))
                x_new_list.extend(x_new_copyr)
        y_new = np.array([class_sample] * len(x_new_list))
        x_new = []
        for item in range(len(x_new_list)):
            x_new.append(x_new_list[item][0])
        return np.array(x_new), y_new

    # 类似于主函数，入口
    def fit_sample(self, x_old, y_old):
        return self.fit(x_old, y_old).sample(x_old, y_old)


if __name__ == '__main__':
    X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                  [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                  [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                  [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                  [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                  [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                  [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                  [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                  [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                  [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
    Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    # 给出的测试方法
    mahakil = MAHAKIL()
    X_resampled, y_resampled = mahakil.fit_sample(X, Y)
    print(X_resampled)
    print(y_resampled)
