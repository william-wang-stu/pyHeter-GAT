# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))
from lib.log import logger
from lib.utils import get_node_types, extend_edges, get_sparse_tensor
import pickle
import numpy as np
import pandas as pd
import os
import psutil
import random
import configparser
from scipy import sparse
from typing import Any, Dict, List
from itertools import combinations, chain
# from numba import njit
# from numba import types
# from numba.typed import Dict, List

config = configparser.ConfigParser()
# NOTE: realpath(__file__)是在获取执行这段代码所属文件的绝对路径, 即~/pyHeter-GAT/src/config.ini
config.read(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'src/config.ini'))
DATA_ROOTPATH = config['DEFAULT']['DataRootPath']
Ntimestage = int(config['DEFAULT']['Ntimestage'])

def save_pickle(obj, filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p']:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    elif ext == '.npy':
        if not isinstance(obj, np.ndarray):
            obj = np.array(obj)
        np.save(filename, obj)
    else:
        pass # raise Error

def load_pickle(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p']:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    elif ext == '.npy':
        return np.load(filename)
    else:
        return None # raise Error

def check_memusage_GB():
    return psutil.Process().memory_info().rss / (1024*1024*1024)

def summarize_distribution(data: List[Any]):
    logger.info(f"list length={len(data)}")
    logger.info("mean list length %.2f", np.mean(data))
    logger.info("max list length %.2f", np.max(data))
    logger.info("min list length %.2f", np.min(data))
    for i in range(1, 20):
        logger.info("%d-th percentile of list length %.2f", i*5, np.percentile(data, i*5))

def wrap(func):
    """
    Usage: wrap(graph.degree)(0)
    """
    def inner(*args, **kwargs):
        # logger.info(f"func={func}, args={args}, kwargs={kwargs}")
        return func(*args, mode="out", **kwargs)
    return inner

def normalize(feat:np.ndarray):
    return feat/(np.linalg.norm(feat,axis=1)+1e-10).reshape(-1,1)

def load_w2v_feature(file, max_idx=0):
    with open(file, "rb") as f:
        nu = 0
        for line in f:
            content = line.strip().split()
            nu += 1
            if nu == 1:
                n, d = int(content[0]), int(content[1])
                # logger.info(f"n={n}, d={d}")
                feature = [[0.] * d for i in range(max(n, max_idx + 1))]
                continue
            index = int(content[0])
            while len(feature) <= index:
                feature.append([0.] * d)
            for i, x in enumerate(content[1:]):
                feature[index][i] = float(x)
    for item in feature:
        assert len(item) == d
    return np.array(feature, dtype=np.float32)

def sample_docs_foreachuser(docs, user_tweet_mp, sample_frac=0.01, min_sample_num=3):
    sample_docs = []
    for user_id in range(len(user_tweet_mp)):
        docs_id = user_tweet_mp[user_id]
        sample_num = int(sample_frac*len(docs_id))
        user_docs = random.choices(docs[docs_id[0]:docs_id[-1]+1], k=max(sample_num, min_sample_num))
        sample_docs.extend(user_docs)
    logger.info(f"sample_docs_num={len(sample_docs)}")
    return sample_docs

def sample_docs_foreachuser2(docs, sample_frac=0.01, min_sample_num=3):
    """
    func: process docs with format of lists of lists, i.e. [[]]
    """
    sample_docs = []
    for texts in docs:
        if len(texts) == 0:
            continue
        sample_num = int(sample_frac*len(texts))
        sample_texts = random.choices(texts, k=max(sample_num, min_sample_num))
        sample_docs.extend(sample_texts)
    logger.info(f"sample_docs_num={len(sample_docs)}")
    return sample_docs
