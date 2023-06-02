#!/usr/bin/env python
# encoding: utf-8
# File Name: data_loader.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/13 16:41
# TODO:

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from lib.log import logger
from utils.utils import load_w2v_feature, load_pickle
from utils.Constants import PAD_WORD, EOS_WORD, PAD, EOS
from utils.graph_aminer import read_user_ids
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from collections import Counter
# from sklearn import preprocessing
import numpy as np
import os
import random
import pickle
import torch
import sklearn
# import itertools
# import igraph

class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

class InfluenceDataSet(Dataset):
    def __init__(self, file_dir, seed, shuffle, model):
        self.graphs = np.load(os.path.join(file_dir, "adjacency_matrix.npy")).astype(np.float32)
        # self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.graphs.shape[1])
        self.graphs += identity
        self.graphs[self.graphs != 0] = 1.0
        if model == "gat":
            self.graphs = self.graphs.astype(np.dtype('B'))
        elif model == "gcn":
            # normalized graph laplacian for GCN: D^{-1/2}AD^{-1/2}
            for i in range(len(self.graphs)):
                graph = self.graphs[i]
                d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
                graph = (graph.T * d_root_inv).T * d_root_inv
                self.graphs[i] = graph
        else:
            raise NotImplementedError
        logger.info("graphs loaded!")

        # wheather a user has been influenced
        # wheather he/she is the ego user
        self.influence_features = np.load(os.path.join(file_dir, "influence_feature.npy")).astype(np.float32)
        logger.info("influence features loaded!")

        self.labels = np.load(os.path.join(file_dir, "label.npy"))
        logger.info("labels loaded!")

        self.vertices = np.load(os.path.join(file_dir, "vertex_id.npy"))
        logger.info("vertex ids loaded!")

        if shuffle:
            self.graphs, self.influence_features, self.labels, self.vertices = \
                    sklearn.utils.shuffle(
                        self.graphs, self.influence_features,
                        self.labels, self.vertices,
                        random_state=seed
                    )

        self.N = self.graphs.shape[0]
        logger.info("%d ego networks loaded, each with size %d" % (self.N, self.graphs.shape[1]))

        n_classes = self.get_num_class()
        class_weight = self.N / (n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)
    
    def get_influence_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.graphs[idx], self.influence_features[idx], self.labels[idx], self.vertices[idx]

class DataConstruct(object):
    ''' For data iteration '''

    def __init__(
            self, dataset_dirpath, batch_size, seed, tmax, num_interval, n_component=None, data_type=0, load_dict=True, shuffle=True, append_EOS=True
        ): # data_type=0(train), =1(valid), =2(test)
        self.batch_size = batch_size
        self.seed = seed
        self.tmax = tmax
        self.num_interval = num_interval
        self.n_component = n_component
        self.data_type = data_type
        self.shuffle = shuffle
        self.append_EOS = append_EOS

        u2idx_filepath, idx2u_filepath = f"{dataset_dirpath}/u2idx.data", f"{dataset_dirpath}/idx2u.data"
        train_data_filepath, valid_data_filepath, test_data_filepath = f"{dataset_dirpath}/train.data", f"{dataset_dirpath}/valid.data", f"{dataset_dirpath}/test.data"
        if not load_dict:
            self._u2idx = {}
            self._idx2u = []
            self._buildIndex(train_data_filepath, valid_data_filepath, test_data_filepath)
            with open(u2idx_filepath, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(idx2u_filepath, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(u2idx_filepath, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(idx2u_filepath, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
            logger.info(f"User Size={self.user_size}")
        
        self._train_data, _train_data_len = self._readCascadeFromFile2(train_data_filepath, min_user=2, max_user=500)
        self._valid_data, _valid_data_len = self._readCascadeFromFile2(valid_data_filepath, min_user=2, max_user=500)
        self._test_data,  _test_data_len  = self._readCascadeFromFile2(test_data_filepath,  min_user=2, max_user=500)
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self._train_data)
        
        if self.data_type == 0:
            self.num_batch = int(np.ceil(len(self._train_data) / self.batch_size))
        elif self.data_type == 1:
            self.num_batch = int(np.ceil(len(self._valid_data) / self.batch_size))
        elif self.data_type == 2:
            self.num_batch = int(np.ceil(len(self._test_data)  / self.batch_size))
        
        self._iter_count = 0
    
    def _buildIndex(self, train_data_filepath, valid_data_filepath, test_data_filepath):
        # compute an index of the users that appear at least once in the training and testing cascades.
        user_set = read_user_ids(train_data_filepath, valid_data_filepath, test_data_filepath)

        pos = 0
        self._u2idx[PAD_WORD] = pos
        self._idx2u.append(PAD_WORD)
        pos += 1

        self._u2idx[EOS_WORD] = pos
        self._idx2u.append(EOS_WORD)
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        
        self.user_size = len(user_set) + 2
        logger.info(f"User Size={self.user_size}")
    
    def _readCascadeFromFile(self, filename, min_user=2, max_user=500):
        """read all cascade from training or testing files. """
        total_len = 0
        cascade_data = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist, tslist = [], []
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])
                    tslist.append(int(float(timestamp)))
            
            if len(userlist) >= min_user and len(userlist) <= max_user:
                total_len += len(userlist)
                if self.append_EOS:
                    userlist.append(EOS)
                    tslist.append(0)
                cascade_data.append({
                    'user': userlist,
                    'ts': tslist,
                })
        return cascade_data, total_len

    def _readCascadeFromFile2(self, filename, min_user=2, max_user=500):
        """read all cascade from training or testing files. """
        per_interval = self.tmax / self.num_interval

        total_len = 0
        cascade_data = []
        data_dict = load_pickle(filename)
        
        key = True if 'user' in list(data_dict.values())[0] else False
        for tag, cascades in data_dict.items():
            userlist = [self._u2idx[elem] for elem in cascades['user' if key else 'seq']]
            tslist = list(cascades['ts' if key else 'interval'])

            intervallist = list(np.ceil((tslist[-1]-np.array(tslist))/(per_interval*3600)))
            for idx, interval in enumerate(intervallist):
                if interval >= self.num_interval:
                    intervallist[idx] = self.num_interval-1
            
            # contentlist = list(cascades['content' if key else 'pre'])

            if self.n_component is not None:
                assert 'label' in cascades
                labellist = list(cascades['label'])
                classid2cnt = Counter(labellist)
                classid = [k for k,c in classid2cnt.most_common(self.n_component) if c>=max(1, int(0.1*len(labellist)))]
            else:
                classid = None

            if len(userlist) >= min_user and len(userlist) <= max_user:
                total_len += len(userlist)
                if self.append_EOS:
                    userlist.append(EOS)
                    tslist.append(0)
                    intervallist.append(0)
                    # contentlist.append(EOS_WORD)
                cascade_data.append({
                    'user': userlist,
                    'ts': tslist,
                    'interval': intervallist,
                    # 'content': contentlist,
                    'classid': classid,
                })
        return cascade_data, total_len
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.num_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''
            max_len = max(len(inst['user']) for inst in insts)
            cascade_users = np.array([
                inst['user'] + [PAD] * (max_len - len(inst['user']))
                for inst in insts])
            cascade_tss = np.array([
                inst['ts'] + [0] * (max_len - len(inst['ts']))
                for inst in insts])
            
            cascade_users_tensor = torch.LongTensor(cascade_users)
            cascade_tss_tensor = torch.LongTensor(cascade_tss)
            return cascade_users_tensor, cascade_tss_tensor

        def pad_to_longest2(insts):
            ''' Pad the instance to the max seq length in batch '''
            max_len = max(len(inst['user']) for inst in insts)
            cascade_users = np.array([
                inst['user'] + [PAD] * (max_len - len(inst['user']))
                for inst in insts])
            cascade_tss = np.array([
                inst['ts'] + [0] * (max_len - len(inst['ts']))
                for inst in insts])
            cascade_intervals = np.array([
                inst['interval'] + [0] * (max_len - len(inst['interval']))
                for inst in insts])
            # cascade_contents = np.array([
            #     inst['content'] + [PAD_WORD] * (max_len - len(inst['content']))
            #     for inst in insts])
            if self.n_component is not None and insts[0]['classid'] is not None:
                cascade_classids = np.array([
                    inst['classid'] + [-1] * (self.n_component - len(inst['classid']))
                    for inst in insts])
                cascade_classids_tensor = torch.LongTensor(cascade_classids)
            else:
                cascade_classids_tensor = None

            cascade_users_tensor = torch.LongTensor(cascade_users)
            cascade_tss_tensor = torch.LongTensor(cascade_tss)
            cascade_intervals_tensor = torch.LongTensor(cascade_intervals)
            # cascade_contents_tensor = torch.LongTensor(cascade_contents)
            return cascade_users_tensor, cascade_tss_tensor, cascade_intervals_tensor, cascade_classids_tensor
        
        if self._iter_count < self.num_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx, end_idx = batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size
            if self.data_type == 0:
                seq_insts = self._train_data[start_idx:end_idx]
            elif self.data_type == 1:
                seq_insts = self._valid_data[start_idx:end_idx]
            elif self.data_type == 2:
                seq_insts = self._test_data[start_idx:end_idx]
                        
            # seq_users, seq_tss = pad_to_longest(seq_insts)
            # return seq_users, seq_tss
            seq_users, seq_tss, seq_intervals, seq_classids = pad_to_longest2(seq_insts)
            return seq_users, seq_tss, seq_intervals, seq_classids
        else:
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(self._train_data)
                        
            self._iter_count = 0
            raise StopIteration()
