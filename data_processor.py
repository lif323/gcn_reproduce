# -*- coding: utf-8 -*-
from cProfile import label
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os
import math

class Dataset:
    # process data
    def __init__(self, args):
        self.dataset_name = args.dataset_str
        print("======================Dataset====================")
    def load_data(self):
        # load the data: x, tx, allx, graph
        # x 训练集
        # tx 测试集
        # allx 所有数据
        # graph 邻接矩阵
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(self.dataset_name, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, y, ty, ally, graph = tuple(objects)

        test_idx_reorder = self.parse_index_file("data/ind.{}.test.index".format(self.dataset_name))
        # test id range [1708, 2707]
        test_idx_range = np.sort(test_idx_reorder)

        if self.dataset_name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            # test_idx_range: 1000 nodex
            # test_idx_range_full: 1015 nodex
            #tx, ty: 1000 * features_num -> tx_extended, ty_extended: 1015 * features_num
            real_tx_num = len(test_idx_reorder)
            full_tx_num = max(test_idx_reorder) - min(test_idx_reorder) + 1
            print("real_tx_num: ", real_tx_num)
            print("full_tx_num: ", full_tx_num)

            tx_extended = sp.lil_matrix((full_tx_num, x.shape[1]))
            # 在正确的位置赋值tx的特征
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended

            # 在正确位置放置 ty标签
            ty_extended = np.zeros((full_tx_num, y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty 
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        # resort test idx base new order why ?????????????????????
        # here is a bug https://github.com/tkipf/gcn/issues/76
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph_new = nx.from_dict_of_lists(graph)

        adj = nx.adjacency_matrix(graph_new)

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        return adj, features, labels,  train_mask, val_mask, test_mask

    def sample_mask(self, idx, n): 
        mask = np.zeros(n)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    
    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        # +-inf的位置设为0
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return self.sparse_to_tuple(features)
    

    def preprocess_adj(self, adj):    
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return self.sparse_to_tuple(adj)

    def sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""
        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col))
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx

    def parse_index_file(self, filename):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

