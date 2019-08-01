#!/usr/bin/env python
# coding: utf-8

import math
import copy
import numpy
from typing import Optional, List
from time import time


class Early_Stopping:

    def __init__(self, max_leaves):
        self.leaf_counter = 0
        self.stop = False
        if max_leaves is None:
            self.incr_counter = lambda: None
            self.check_early_stopping = lambda: False
        else:
            self.max = max_leaves
            self.incr_counter = self._incr_counter
            self.check_early_stopping = self._check_early_stopping

    def _check_early_stopping(self):
        return self.stop

    def _incr_counter(self):
        self.leaf_counter += 1
        if self.leaf_counter == self.max:
            self.stop = True


class Neighbours():
    """ Class that can be used to keep track of the 
    k nearest neighbours computed. It also conducts
    the "brute-force" phase that takes place in the
    leaves of a k-d tree (see function 'add').
    """

    def __init__(self, k):
        self.k = k

        self._neighbors = [(None, None, float("inf")) for i in range(self.k)]

    def get_dists_indices(self):
        indices = [neigh[1] for neigh in self._neighbors]
        dists = [neigh[2] for neigh in self._neighbors]

        return numpy.array(dists), numpy.array(indices)

    def add(self, points, indices, query):
        for i in range(len(points)):
            p = points[i]
            idx = indices[i]
            dist = self._distance(p, query)

            self._neighbors.append([p, idx, dist])
            self._neighbors = sorted(self._neighbors, key=lambda n: n[2])
            self._neighbors = self._neighbors[:self.k]

    def get_max_dist(self):
        return self._neighbors[-1][2]

    def _distance(self, x, y):
        dist = ((x - y) ** 2).sum()
        return math.sqrt(dist)


class Node():
    """ Class that represents a single node of
    a k-d tree. If both 'left' and 'right' are 
    None, then the node is a leaf. The local 
    variables 'points' and 'indices' are used
    to store the points/indices assigned to a 
    leaf.
    
    Otherwise, it is an internal node that 
    stores the median (splitting hyperplane)
    """

    def __init__(self,
                 left,
                 right,
                 median=None,
                 points=None,
                 indices=None):
        self.left = left
        self.right = right
        self.median = median
        self.points = points
        self.indices = indices


class KDTree():

    def __init__(self, leaf_size=30):
        """ Instantiates a k-d tree.
        
        Parameters
        ----------
        leaf_size : int, default 30
            The leaf size, i.e., the maximal 
            number of points stored in a leaf 
            of the k-d tree.
        """

        self.leaf_size = leaf_size

    def fit(self, X):
        """
        
        Parameters
        ----------
        X : array-like of shape (n, d)
            A Numpy array containing n data 
            points each having d features                
        """

        # remember dimension for which the tree was built
        self._dim = len(X[0])

        # generate a list of the "original" indices that
        # are processed in a similar way as the points; 
        # this is needed in order to obtain the indices
        # of the neighbours compuated for a query.
        original_indices = numpy.array(range(len(X)))

        # build tree recursively
        self._root = self._build_tree(copy.deepcopy(X),
                                      original_indices,
                                      depth=0)

    def query(self, X, k=1, max_leaves=None, alpha=1.0):
        """ Computes the k nearest neighbors for each 
        point in X.
        
        Parameters
        ----------
        X : array-like of shape (n, d)
            A Numpy array containing n data 
            points each having d features
        k : int, default 1
            The number of nearest neighbours to 
            be computed
            
        Returns
        -------
        dists, indices : arrays of shape (n, k)
            Two arrays containing, for each query point,
            the distances and the associated indices of
            its k nearest neighbors w.r.t. the points
            used for building the tree.
        """

        if self._root is None:
            raise Exception("Tree not fitted yet!")

        if len(X[0]) != self._dim:
            raise Exception("Tree was fitted for points of dimension: {}".format(self._dim))

        # initialize two empty arrays that will be used to 
        # store the distances and the associated indices
        dists = numpy.empty((len(X), k), dtype=numpy.float64)
        indices = numpy.empty((len(X), k), dtype=numpy.int32)

        # iterate over all query points
        n_points = len(X)
        for i in range(len(X)):
            if i % 1000 == 0:
                print('evaluating point {} of {}'.format(i, n_points))
            # initialize the neighbours object, which
            # will keep track of the nearest neighbours
            neighbours = Neighbours(k)

            # start recursive search
            self._recursive_search(self._root,
                                   X[i],
                                   k,
                                   depth=0,
                                   neighbours=neighbours, early_stopping=Early_Stopping(max_leaves),
                                   alpha=alpha)

            # get the final distances and indices for 
            # the current query and store them at 
            # position i in the arrays dists and indices 
            dists_query, indices_query = neighbours.get_dists_indices()
            dists[i, :] = dists_query
            indices[i, :] = indices_query

        return dists, indices

    def _build_tree(self, pts, indices, depth):
        """ Builds a k-d tree for the points given in pts. Since
        we are also interested in the indidces afterwards, we also
        keep track of the (original) indices.
        
        This code is similar to the pseudocode given on 
        slides 27-29 of L3_LSDA.pdf
        """

        # if only self.leaf_size points are left, stop
        # the recursion and generate a leaf node
        if len(pts) <= self.leaf_size:
            return Node(left=None,
                        right=None,
                        points=pts,
                        indices=indices)

        # select axis
        axis = depth % self._dim

        # sort the points w.r.t. dimension 'axis';
        # also sort the indices accordingly
        partition = pts[:, axis].argsort()
        pts = pts[partition]
        indices = indices[partition]

        # compute splitting index and median value
        split_idx = math.floor(len(pts) / 2)
        if len(pts) % 2 == 1:
            median = pts[split_idx, axis]
        else:
            median = 0.5 * (pts[split_idx, axis] + pts[split_idx + 1, axis])

        # build trees for children recursively ...
        lefttree = self._build_tree(pts[:split_idx, :], indices[:split_idx], depth + 1)
        righttree = self._build_tree(pts[split_idx:, :], indices[split_idx:], depth + 1)

        # return node storing all the relevant information
        return Node(left=lefttree, right=righttree, median=median)

    def _recursive_search(self, node, query, k, depth, neighbours, early_stopping, alpha):
        if early_stopping.check_early_stopping():
            return

        if (node.left == None and node.right == None):
            neighbours.add(node.points, node.indices, query)
            early_stopping.incr_counter()
            return

        # axis to be checked (same order as during construction)
        axis = depth % self._dim

        # select next subtree candidate
        if query[axis] < node.median:
            first = node.left
            second = node.right
        else:
            first = node.right
            second = node.left

        # check first subtree
        self._recursive_search(first, query, k, depth + 1, neighbours, early_stopping, alpha)

        # while going up again (to the root): check if we 
        # still have to search in the second subtree! 
        if abs(node.median - query[axis]) < neighbours.get_max_dist() / alpha:
            self._recursive_search(second, query, k, depth + 1, neighbours, early_stopping, alpha)


class NearestNeighborClassifier(object):

    def __init__(self, n_neighbors=3, leaf_size=20):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

        # Added
        self.Tree = KDTree(leaf_size=leaf_size)
        self.y = None

    def fit(self, X, y):
        self.Tree.fit(X)
        self.y = y

    def _query(self, X, max_leaves=None, alpha=1.0):
        dists, indss = self.Tree.query(X, k=self.n_neighbors, max_leaves=max_leaves, alpha=alpha)
        return indss

    def predict(self, X, max_leaves=None, alpha=1.0):
        indss = self._query(X, max_leaves=max_leaves, alpha=alpha)
        preds = [round(sum([self.y[idx] for idx in inds]) / self.n_neighbors) for inds in indss]
        return preds


from General.Evaluation_Model import Evaluation_Model
from General.Conventions import Load_Main_Dataset
from _0_DataCreation.Raw_Data_Transformations import df_to_row, data_to_keep
import pandas as pd
import numpy as np


class KNN_org_data_post_pca(Evaluation_Model):

    def __init__(self, n_neighbors: int = 3, leaf_size: int = 20, max_leaves: Optional[int] = None,
                 pruning: float = 1.0, max_rows=-1, nrow: int = 10, cols_to_keep: Optional[List] = None):

        if cols_to_keep is None:
            self.cols_to_keep = list(range(16))
        else:
            self.cols_to_keep = cols_to_keep

        self.max_rows = max_rows
        self.nrow = nrow
        self.max_leaves = max_leaves
        self.pruning = pruning
        self.model = NearestNeighborClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
        super(KNN_org_data_post_pca, self).__init__(name=self.__class__.__name__,
                                                    reproducible_arguments=dict(n_neighbors=n_neighbors,
                                                                                leaf_size=leaf_size,
                                                                                max_leaves=max_leaves, pruning=pruning,
                                                                                nrow=nrow, cols_to_keep=cols_to_keep,
                                                                                max_rows=max_rows))

    def _Fit(self, Train_Datasets):
        flat_data = []
        labels = []
        i = 0
        start = time()
        for ds in Train_Datasets:
            for id, label, data in Load_Main_Dataset(Dataset=ds, max_row=self.max_rows):
                labels.append(label)
                flat_data.append(df_to_row(data_to_keep(data, nrow=self.nrow, cols_to_keep=self.cols_to_keep)))
                i += 1
                if i % 1000 == 0:
                    print('fit_progress: {}, time: {}'.format(i, time() - start))
                    start = time()
        self.model.fit(pd.concat(flat_data).values, np.asarray(labels))

    def _Predict(self, Test_Datasets):
        flat_data = []
        i = 0
        start = time()
        for ds in Test_Datasets:
            for id, label, data in Load_Main_Dataset(Dataset=ds, max_row=-1):
                flat_data.append(df_to_row(data_to_keep(data, nrow=self.nrow, cols_to_keep=self.cols_to_keep)))
                i += 1
                if i % 1000 == 0:
                    print('pred_progress: {}, time: {}'.format(i, time()-start))
                    start = time()
        preds = self.model.predict(pd.concat(flat_data).values, max_leaves=self.max_leaves, alpha=self.pruning)
        return preds


def time_testing(**kwargs):
    from General.Conventions import Datasets
    m = KNN_org_data_post_pca(**kwargs)
    fit_start = time()
    m._Fit([Datasets.fold2])
    print('fit_done')
    fit_end = time()
    # print(m._Predict([Datasets.fold1, Datasets.fold3]))
    m._Predict([Datasets.fold1])
    predict_end = time()
    ret = 'fitting time: {}, predict_time: {}'.format(fit_end - fit_start, predict_end - fit_end)
    print(ret)
    return ret


def Check_Score(**kwargs):
    from General.Conventions import Datasets
    m = KNN_org_data_post_pca(**kwargs)
    start = time()
    result = m._Evaluation_Model__Fit_and_Score(Train_Datasets=[Datasets.fold2], Test_Datasets=[Datasets.fold1])
    print(time()-start)
    print(result)
    return result


def Reports():
    ks = [6]
    # max_leavess = [10*(i+1) for i in range(2)]
    max_leavess = [20]
    nrows = [1, 10]
    results = []
    for k in ks:
        for max_leaves in max_leavess:
            for nrow in nrows:
                try:

                    m = KNN_org_data_post_pca(n_neighbors=k, max_leaves=max_leaves, nrow=nrow)
                    results.append(m.Report_Model(Full=False)[-1])
                except:
                    results.append(-1)


    try:
        m = KNN_org_data_post_pca(nrow=30)  # Full model
        results.append(m.Report_Model(Full=False)[-1])

    except:
        results.append(-1)

    print(results)
    return results


if __name__ == '__main__':
    # time_testing(n_neighbors=3, leaf_size=20, max_leaves=None, pruning=1.0)
    # time_testing(n_neighbors=6, leaf_size=20, max_leaves=None, pruning=1.0)
    # time_testing(n_neighbors=3, leaf_size=100, max_leaves=None, pruning=1.0)
    # time_testing(n_neighbors=3, leaf_size=20, max_leaves=20, pruning=1.0, nrow=10)  # by far most impactful
    # time_testing(n_neighbors=3, leaf_size=20, max_leaves=None, pruning=2.0)

    # Check_Score(n_neighbors=3, leaf_size=20, max_leaves=20, pruning=1.0, nrow=1)
    # Check_Score(n_neighbors=3, leaf_size=20, max_leaves=20, pruning=1.0, nrow=10)

    if 0:
        x = [time_testing(max_leaves=10, nrow=20, max_rows=10**1),
        time_testing(max_leaves=10, nrow=20, max_rows=10**2),
        time_testing(max_leaves=10, nrow=20, max_rows=5*(10**2)),
        time_testing(max_leaves=10, nrow=20, max_rows=1*(10**3)),
        time_testing(max_leaves=10, nrow=20, max_rows=5*(10**3)),
        time_testing(max_leaves=10, nrow=20, max_rows=1*(10**4)),
        time_testing(max_leaves=10, nrow=20, max_rows=5*(10**4)),
        time_testing(max_leaves=10, nrow=20, max_rows=-1)]

        for l in x:
            print(l)

        # 20k preds

        # fitting time: 0.05886983871459961, predict_time: 72.83470416069031
        # fitting time: 0.412686824798584, predict_time: 83.91665077209473
        # fitting time: 1.805863857269287, predict_time: 94.81306886672974
        # fitting time: 3.3680343627929688, predict_time: 91.53443002700806
        # fitting time: 16.323052883148193, predict_time: 99.00771355628967
        # fitting time: 32.93215775489807, predict_time: 103.88673305511475
        # fitting time: 165.54533290863037, predict_time: 91.85432243347168
        # fitting time: 322.3274636268616, predict_time: 85.63776421546936
    if 0:
        x = [time_testing(max_leaves=10, nrow=20, max_rows=-1), # fitting time: 324.6860132217407, predict_time: 333.8117768764496
             time_testing(max_leaves=20, nrow=20, max_rows=-1)], # fitting time: 312.39713191986084, predict_time: 405.4976382255554

        for l in x:
            print(l)

    if 0:
        print([Check_Score(max_leaves=20, nrow=20, max_rows=-1)])

    if 0:
        n_runs = 10
        total_run_time = 0
        for _ in range(n_runs):
            start = time()
            model._query(X_test)
            total_run_time += time() - start
        print("Time to query (avg of {} runs): {}".format(n_runs, total_run_time / n_runs))

    if 0:
        Reports()