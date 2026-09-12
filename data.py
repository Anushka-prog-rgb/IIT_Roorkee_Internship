import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Multi_view_data(Dataset):
    """
    Load multi-view data. Supports three on-disk .mat layouts:

    - Pre-split (e.g. handwritten_6views): keys x{v}_train / x{v}_test, each
      (n_samples, n_features), plus gt_train / gt_test (n_samples, 1).
    - Unsplit, per-view keys (e.g. scene15): keys X{v} (capital X), each
      stored transposed as (n_features, n_samples), plus a single gt
      (n_samples, 1).
    - Unsplit, cell array (most datasets, e.g. NUS-WIDE, Cora, Reuters-1500):
      a single 'X' key holding an (n_views, 1) (or (1, n_views)) object
      array whose cells are already (n_samples, n_features) -- occasionally
      as scipy.sparse matrices (e.g. Reuters-1500) -- plus a single 'y'
      (n_samples, 1). Labels may be 0- or 1-indexed depending on the file.

    Neither unsplit layout ships a train/test split, so one is carved out
    here with a fixed random_state (stratified, so train=True/False calls
    agree on the partition), and the MinMax scaler is fit on the train split
    only and reused on test to avoid leaking test statistics into the
    normalization.
    """

    def __init__(self, root, train=True, test_size=0.2, random_state=0):
        """
        :param root: data name and path (without .mat extension)
        :param train: load training set or test set
        :param test_size: held-out fraction, only used for unsplit layouts
        :param random_state: split seed, only used for unsplit layouts
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)
        keys = set(dataset.keys())
        self.X = dict()

        if 'gt_train' in keys or 'gt_test' in keys:
            view_number = int((len(dataset) - 5) / 2)
            if train:
                for v_num in range(view_number):
                    self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])
                y = dataset['gt_train']
            else:
                for v_num in range(view_number):
                    self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
                y = dataset['gt_test']
        else:
            if 'X1' in keys:
                # per-view keys, each stored transposed (n_features, n_samples)
                view_number = 0
                while ('X%d' % (view_number + 1)) in keys:
                    view_number += 1
                views = [_to_dense(dataset['X%d' % (v + 1)]).T for v in range(view_number)]
                y_all = dataset['gt'].reshape(-1)
            else:
                # single cell array 'X', cells already (n_samples, n_features)
                cells = dataset['X'].reshape(-1)
                views = [_to_dense(cell) for cell in cells]
                view_number = len(views)
                y_all = dataset['y'].reshape(-1)

            n = y_all.shape[0]
            train_idx, test_idx = train_test_split(
                np.arange(n), test_size=test_size, random_state=random_state, stratify=y_all)
            split_idx = train_idx if train else test_idx

            for v_num in range(view_number):
                scaler = MinMaxScaler((0, 1))
                scaler.fit(views[v_num][train_idx])
                self.X[v_num] = scaler.transform(views[v_num][split_idx]).astype(np.float32)
            y = y_all[split_idx]

        if np.min(y) >= 1:
            y = y - np.min(y)
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])


def _to_dense(x):
    """Densify a view matrix (some datasets, e.g. Reuters-1500, store views
    as scipy.sparse matrices, which MinMaxScaler can't fit/transform)."""
    if sp.issparse(x):
        return np.asarray(x.todense())
    return x


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x
