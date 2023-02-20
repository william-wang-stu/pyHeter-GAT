# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))
from lib.log import logger
from lib.utils import get_node_types, extend_edges, get_sparse_tensor
import pickle
import numpy as np
from typing import Any, Dict, List
import igraph
import os
import random
import psutil
import datetime
import torch
from tensorboard_logger import tensorboard_logger
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from scipy import sparse
from scipy.sparse import csr_matrix
import copy
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn

config = configparser.ConfigParser()
# NOTE: realpath(__file__)是在获取执行这段代码所属文件的绝对路径, 即~/pyHeter-GAT/src/utils.py
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini'))
DATA_ROOTPATH = config['DEFAULT']['DataRootPath']
Ntimestage = int(config['DEFAULT']['Ntimestage'])

class SubGraphSample:
    # NOTE: "Mutable Default Arguments": 
    # 函数的缺省参数是函数定义时作为`__default__`属性附着于函数这个对象上的, 
    # 而如果缺省参数是如list的可变对象, 那么每次调用都会改变它, 也即缺省参数不是每次调用时初始化的, 而是仅在定义时初始化一次
    """
    def foo(a=[]):
        a.append(5)
        return a
    >>> foo()
    [5]
    >>> foo()
    [5, 5]
    >>> foo()
    [5, 5, 5]
    """
    def __init__(self, adj_matrices=None, influence_features=None, vertex_ids=None, labels=None, tags=None, time_stages=None):
        arguments = locals()
        for key, value in arguments.items():
            arguments[key] = [] if value is None else value

        self.adj_matrices = arguments["adj_matrices"]
        self.influence_features = arguments["influence_features"]
        self.vertex_ids = arguments["vertex_ids"]
        self.labels = arguments["labels"]
        self.tags = arguments["tags"]
        self.time_stages = arguments["time_stages"]
    def __len__(self):
        return len(self.labels)

class HeterSubGraphSample:
    def __init__(self, heter_adj_matrices=None, initial_features=None, vertex_ids=None, labels=None):
        arguments = locals()
        for key, value in arguments.items():
            arguments[key] = [] if value is None else value
        
        self.heter_adj_matrices = arguments["heter_adj_matrices"]
        self.initial_features = arguments["initial_features"]
        self.vertex_ids = arguments["vertex_ids"]
        self.labels = arguments["labels"]
    def __len__(self):
        return len(self.labels)

class ChunkSampler(Sampler):
    """
    Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0, gap=1):
        self.num_samples = num_samples
        self.start = start
        self.gap = gap

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples, self.gap))

    def __len__(self):
        return self.num_samples // self.gap

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def check_memusage_GB():
    return psutil.Process().memory_info().rss / (1024*1024*1024)

def summarize_distribution(data: List[Any]):
    logger.info(f"list length={len(data)}")
    logger.info("mean list length %.2f", np.mean(data))
    logger.info("max list length %.2f", np.max(data))
    logger.info("min list length %.2f", np.min(data))
    for i in range(1, 20):
        logger.info("%d-th percentile of list length %.2f", i*5, np.percentile(data, i*5))

def init_graph(nb_nodes:int, edgelist:List[Any], is_directed:bool=False, outputfile_dirpath:str="", save_graph:bool=False):
    graph = igraph.Graph(nb_nodes, directed=is_directed)
    graph.add_edges(edgelist)
    if not is_directed:
        graph.to_undirected()
    graph.simplify()

    if save_graph:
        # Save Graph in IGraph Format
        os.makedirs(outputfile_dirpath, exist_ok=True)
        # with open(os.path.join(outputfile_dirpath, "igraph_edgelist"), "w") as f:
        #     graph.write(f, format="edgelist")
        save_pickle(graph, "{}/igraph-{}.p".format(outputfile_dirpath, "directed" if is_directed else "undirected"))
        logger.info("Save Network in IGraph Format to {}/igraph-{}.p".format(outputfile_dirpath, "directed" if is_directed else "undirected"))

    return graph

def gen_subgraph(graph_dirpath:str, nb_users:int, subgraph_dirpath:str):
    nodes = {
        "User": load_pickle(os.path.join(graph_dirpath, "Users.p")),
        "Tweet": load_pickle(os.path.join(graph_dirpath, "Tweets.p")),
    }
    edges = {
        "U-U": load_pickle(os.path.join(graph_dirpath, "U-U.p")),
        "U-T": load_pickle(os.path.join(graph_dirpath, "U-T.p")),
    }

    sample_nodes = {"User": {}, "Tweet": {}}
    sample_edges = {"U-U":  [], "U-T":   []}
    os.makedirs(subgraph_dirpath, exist_ok=True)
    
    # Sample User Nodes
    for user_id, graphid in nodes["User"].items():
        if graphid < nb_users:
            sample_nodes["User"][user_id] = graphid
    # save_pickle(sample_nodes["User"], os.path.join(subgraph_dirpath, f"Users_u{nb_users}.p"))
    save_pickle(sample_nodes["User"], os.path.join(subgraph_dirpath, f"Users.p"))
    
    # Sample User-User Edges
    for user1, user2 in edges["U-U"]:
        if user1 < nb_users and user2 < nb_users:
            sample_edges["U-U"].append((user1, user2))
    save_pickle(sample_edges["U-U"], os.path.join(subgraph_dirpath, f"U-U.p"))

    # Sample User-Tweet Edges
    for user_id, tweet_id in edges["U-T"]:
        if user_id < nb_users:
            sample_edges["U-T"].append((user_id, tweet_id))
    save_pickle(sample_edges["U-T"], os.path.join(subgraph_dirpath, f"U-T.p"))

    # Sample Tweet Nodes
    nb_tweets = len(sample_edges["U-T"])
    for tweet_id, graphid in nodes["Tweet"].items():
        if graphid < nb_tweets:
            sample_nodes["Tweet"][tweet_id] = graphid
    save_pickle(sample_nodes["Tweet"], os.path.join(subgraph_dirpath, f"Tweets.p"))

    logger.info(f"Nodes Info: Users={nb_users}, Tweets={nb_tweets}, Total={nb_users+nb_tweets}")
    logger.info("Edges Info: U-U={}, U-T={}, Total={}".format(len(sample_edges["U-U"]), len(sample_edges["U-T"]), len(sample_edges["U-U"])+len(sample_edges["U-T"])))

def gen_subactionlog(actionlog_dirpath:str, nb_users:int, subactionlog_dirpath:str):
    actionlog = load_pickle(os.path.join(actionlog_dirpath, "ActionLog.p"))
    subactionlog = {}

    for hashtag, values in actionlog.items():
        subvalues = []
        for graphid, timestamp in values:
            if graphid < nb_users:
                subvalues.append((graphid, timestamp))
        subactionlog[hashtag] = subvalues
    save_pickle(subactionlog, os.path.join(subactionlog_dirpath, f"ActionLog_u{nb_users}.p"))

def gen_user_sets(actionlog, Ntimestages:int, utedges_dirpath: str):
    user_sets = [[set() for _ in range(len(actionlog))] for _ in range(Ntimestages)]

    for hidx, (hashtag, values) in enumerate(actionlog.items()):
        if len(values) == 0:
            continue
        lower_b, upper_b = values[0][1], values[-1][1]
        time_span = (upper_b-lower_b)/(Ntimestages+1)
        elem_idx = 0
        for tidx in range(8):
            start_t, end_t = lower_b+tidx*time_span, lower_b+(tidx+1)*time_span
            while elem_idx < len(values) and values[elem_idx][1] <= end_t:
                user_sets[tidx][hidx].add(values[elem_idx][0])
                elem_idx += 1

    logger.info("User_Sets Num: {}".format(
        " ".join([f"t{tidx}={sum([len(elem) for elem in user_sets[tidx]])}" for tidx in range(Ntimestages)])
    ))
    save_pickle(user_sets, os.path.join(utedges_dirpath, "user_sets.p"))

def add_utedges(data_dirpath:str, utedges_dirpath:str, Ntimestages:int=8):
    # 0. 
    actionlog   = load_pickle(os.path.join(data_dirpath, "ActionLog.p"))
    tweetid2tag = load_pickle(os.path.join(data_dirpath, "text/tweetid2tag.p"))
    node_stat   = load_pickle(os.path.join(data_dirpath, "Node-Stat.p"))
    utedges     = load_pickle(os.path.join(data_dirpath, "graph/U-T.p"))
    tweets      = load_pickle(os.path.join(data_dirpath, "graph/Tweets.p"))
    tweetid2graphid = {value:key for key,value in tweets.items()}
    logger.info("Finish Loading Data...")

    # 1. Get User_Sets Per Time Stage Per Hashtag, i.e. user_sets[tidx][hidx]
    os.makedirs(utedges_dirpath, exist_ok=True)
    if not os.path.exists(os.path.join(utedges_dirpath, "user_sets.p")):
        gen_user_sets(actionlog=actionlog, Ntimestages=Ntimestages, utedges_dirpath=utedges_dirpath)
    user_sets = load_pickle(os.path.join(utedges_dirpath, "user_sets.p"))

    # 2. Gen U-T Edges
    sampled_length = [[] for _ in range(Ntimestages)]
    for tidx in range(Ntimestages):
        ut_edges = []
        for eidx, (user_id, tweet_id) in enumerate(utedges):
            # 1. Get Hashtag which tweet_id belongs to
            # 2. Get Other Users within same hashtag and time-stage
            # 3. Build Additional U-T Edges between current tweet_id and other-users
            if eidx % 10000000 == 0:
                logger.info(f"Enumerating UT-Edges: eidx={eidx}")
            tag = tweetid2tag[int(tweetid2graphid[tweet_id])]
            involved_users = user_sets[tidx][tag]
            if user_id not in involved_users:
                # logger.info(f"Error: user={user_id} not in involved_users={involved_users}")
                continue
            involved_users = [user_id] + random.choices(list(involved_users), k=min(node_stat[user_id]["degree"], node_stat[user_id]["text_num"], len(involved_users)))
            sampled_length[tidx].append(len(involved_users))
            for in_user in involved_users:
                ut_edges.append((in_user, tweet_id))
        logger.info(f"Additional UT-Edges Num={len(ut_edges)}")
        save_pickle(ut_edges, os.path.join(utedges_dirpath, f"ut_edges_t{tidx}.p"))

    # 3. Analyze Distribution of Sampled Involved_Users
    for tidx in range(Ntimestages):
        logger.info(f"Analyzing Distribution of Sampled Involved_Users in Time Stage={tidx}")
        summarize_distribution(sampled_length[tidx])

def build_matrices(graph_dirpath:str, matrices_filepath:str):
    nodes = {
        "User": load_pickle(os.path.join(graph_dirpath, "Users.p")),
        "Tweet": load_pickle(os.path.join(graph_dirpath, "Tweets.p")),
    }
    nodes["ALL"] = nodes["User"] | nodes["Tweet"]
    edges = {
        "U-U": load_pickle(os.path.join(graph_dirpath, "U-U.p")),
        # "U-T": load_pickle(os.path.join(graph_dirpath, "U-T.p")),
        "U-T": load_pickle("/root/Heter-GAT/src/add_utedges_u20000/ut_edges_t0.p")
    }

    start_idx_mp = {}
    indices = 0
    for node_type in ["User", "Tweet"]:
        start_idx_mp[node_type] = indices
        indices += len(nodes[node_type])
    logger.info(f"indices={indices}, start_idx_mp={start_idx_mp}")

    matrices = {"ALL": []}
    for edge_type in ["U-U", "U-T"]:
        node1_t, node2_t = get_node_types(edge_type)
        extended_edges = edges[edge_type]
        if start_idx_mp[node1_t] or start_idx_mp[node2_t]:
            # logger.info(f"node1_t={node1_t}, node2_t={node2_t}, start_idx_node1_t={start_idx_mp[node1_t]}, start_idx_node2_t={start_idx_mp[node2_t]}")
            extended_edges = extend_edges(extended_edges, start_idx_mp[node1_t], start_idx_mp[node2_t])
        matrices[edge_type] = extended_edges
        
        # accumulate all edgelists
        matrices["ALL"] += matrices[edge_type]

    logger.info("Nodes Info: {}".format(" ".join(f"num_{node_t}={len(nodes[node_t])}" for node_t in ["User", "Tweet", "ALL"])))
    logger.info("Edgelists Info: {}".format(" ".join(f"num_{key}={len(value)}" for key, value in matrices.items())))
    os.makedirs(matrices_filepath, exist_ok=True)
    save_pickle(matrices, os.path.join(matrices_filepath, "Edgelist-Matrices.p"))

# Usage: wrap(graph.degree)(0)
def wrap(func):
    def inner(*args, **kwargs):
        # logger.info(f"func={func}, args={args}, kwargs={kwargs}")
        return func(*args, mode="out", **kwargs)
    return inner

def set_ar_values(ar, indices, value):
    new_ar = []
    for idx, elem in enumerate(ar):
        new_ar.append(value if idx in indices else elem)
    return new_ar

def find_rt_bound(elem, bound_ar):
    idx = 0
    for rt_bound in bound_ar:
        if elem < rt_bound:
            break
        idx += 1
    return idx if idx<len(bound_ar) else len(bound_ar)-1

def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    uugraph_batch, utgraph_batch, labels_batch, feats_batch = zip(*batch)
    if type(uugraph_batch[0]) == csr_matrix:
        uugraph_batch = torch.stack([get_sparse_tensor(uugraph.tocoo()) for uugraph in uugraph_batch])
    else:
        uugraph_batch = torch.FloatTensor(uugraph_batch)

    if type(utgraph_batch[0]) == csr_matrix:
        utgraph_batch = torch.stack([get_sparse_tensor(utgraph.tocoo()) for utgraph in utgraph_batch])
    else:
        utgraph_batch = torch.FloatTensor(utgraph_batch)
    
    if type(labels_batch[0]).__module__ == 'numpy':
        # NOTE: https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        labels_batch = torch.LongTensor(labels_batch)
    
    if type(feats_batch[0]).__module__ == 'numpy':
        feats_batch = torch.FloatTensor(np.array(feats_batch))
    return uugraph_batch, utgraph_batch, labels_batch, feats_batch

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

def sample_tweets_around_user(users:set, ut_mp:dict, tweets_per_user:int, counts_matter:bool=False, return_edges:bool=False):
    """
    功能: 根据ut_mp为users中的每个user挑选min(tweets_per_user, len(ut_mp[user]))个推文邻居节点
    参数: 
        counts_matter表示总共挑选的推文节点数量必须等于len(users)*tweets_per_user,不满足则返回空推文节点集合
    返回:
        tweet_nodes(set), enough_tweet_nodes(bool,counts_matter=True), ut_edges(list,return_edges=True)
    """
    tweets, remaining_tweets, ut_edges = set(), set(), set()
    for user in users:
        selected_tweets = random.choices(ut_mp[user], k=min(len(ut_mp[user]), tweets_per_user))
        tweets |= set(selected_tweets)
        if counts_matter:
            remaining_tweets |= set(ut_mp[user]) - set(selected_tweets)
        if return_edges:
            for tweet in selected_tweets:
                ut_edges.add((user, tweet))
    
    return tweets, remaining_tweets, list(ut_edges)

def gen_random_tweet_ids(samples: SubGraphSample, outdir: str, tweets_per_user:int=5):
    tweet_ids = []
    sample_ids = []
    ut_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))

    # TODO: 重构
    for idx in range(len(samples.labels)):
        if idx and idx % 10000 == 0:
            logger.info(f"idx={idx}, sample_ids={len(sample_ids)}, tweet_ids={len(tweet_ids)}")
        stage = samples.time_stages[idx]
        selected_tweet_ids  = set()
        candidate_tweet_ids = set()
        for vertex_id in samples.vertex_ids[idx]:
            available_tweet_ids = ut_mp[stage][vertex_id]
            random_ids = np.random.choice(available_tweet_ids, size=min(tweets_per_user, len(available_tweet_ids)), replace=False)
            selected_tweet_ids  |= set(random_ids)
            candidate_tweet_ids |= set(available_tweet_ids)-set(random_ids)
        candidate_tweet_ids -= selected_tweet_ids
        # logger.info(f"Length: sample={len(selected_tweet_ids)}, remain={len(candidate_tweet_ids)}, expected={len(samples.vertex_ids[idx])*tweets_per_user}")

        if len(selected_tweet_ids) != len(samples.vertex_ids[idx])*tweets_per_user:
            diff = len(samples.vertex_ids[idx])*tweets_per_user - len(selected_tweet_ids)
            if diff > len(candidate_tweet_ids):
                continue
            selected_tweet_ids |= set(np.random.choice(list(candidate_tweet_ids), size=diff, replace=False))
        sample_ids.append(idx)
        tweet_ids.append(selected_tweet_ids)
    logger.info(f"Finish Sampling Random Tweets... sample_ids={len(sample_ids)}, tweet_ids={len(tweet_ids)}")

    os.makedirs(outdir, exist_ok=True)
    selected_samples = SubGraphSample(
        adj_matrices=samples.adj_matrices[sample_ids],
        influence_features=samples.influence_features[sample_ids],
        vertex_ids=samples.vertex_ids[sample_ids],
        labels=samples.labels[sample_ids],
        tags=samples.tags[sample_ids],
        time_stages=samples.time_stages[sample_ids],
    )
    save_pickle(sample_ids, os.path.join(outdir, "sample_ids.p"))
    save_pickle(tweet_ids, os.path.join(outdir, "tweet_ids.p"))
    save_pickle(selected_samples, os.path.join(outdir, "selected_samples.p"))
    logger.info("Finish Saving pkl...")

def extend_subnetwork(file_dir: str):
    hs_filedir = os.path.join(DATA_ROOTPATH, file_dir).replace('stages_', 'hs_')
    samples = load_pickle(os.path.join(hs_filedir, "selected_samples.p"))
    tweet_ids = load_pickle(os.path.join(hs_filedir, "tweet_ids.p"))
    assert len(samples) == len(tweet_ids)

    tweetid2userid_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/tweetid2userid_mp.p"))
    vertex_ids = samples.vertex_ids
    adjs       = samples.adj_matrices
    adjs[adjs != 0] = 1.0
    adjs = adjs.astype(np.dtype('B'))

    extended_vertices, extended_adjs = [], []
    for idx in range(len(samples)):
        subnetwork = np.array(np.concatenate((vertex_ids[idx], np.array(list(tweet_ids[idx])))), dtype=int)
        extended_vertices.append(subnetwork)

        subnetwork_size, num_users = len(subnetwork), len(vertex_ids[idx])
        elem_idx_mp = {elem:idx for idx,elem in enumerate(subnetwork)}
        uu_adj = np.array([[0]*subnetwork_size for _ in range(subnetwork_size)], dtype='B')
        uu_adj[:num_users,:num_users] = adjs[idx]
        # NOTE: Get Corresponding User_id By Tweet_id, and then convert them into indexes in extend_subnetwork
        ut_adj = copy.deepcopy(uu_adj)
        for tweet_id in tweet_ids[idx]:
            user_id = tweetid2userid_mp[tweet_id]
            net_userid = elem_idx_mp[user_id]
            net_tweetid = elem_idx_mp[tweet_id]
            ut_adj[net_userid][net_tweetid] = 1
        extended_adjs.append([uu_adj, ut_adj])
    extended_vertices, extended_adjs = np.array(extended_vertices), np.array(extended_adjs)
    save_pickle(extended_vertices, os.path.join(hs_filedir, "extended_vertices.p"))
    save_pickle(extended_adjs, os.path.join(hs_filedir, "extended_adjs.p"))

def reindex_graph(old_nodes:list, old_edges:list, add_self_loop:bool=True):
    nodes = {}
    node_indices, max_indices = 0, 0
    for nodes_l in old_nodes:
        for node in nodes_l:
            nodes[node+max_indices] = node_indices
            node_indices += 1
        max_indices = max(list(nodes.keys()))+1
    
    edges = [[], []]
    for from_, to_ in old_edges[0]:
        edges[0].append((from_, to_))
    offset = max(old_nodes[0])+1
    for from_, to_ in old_edges[1]:
        edges[1].append((nodes[from_], nodes[to_+offset]))
    
    if add_self_loop:
        for node in range(len(old_nodes[0])):
            edges[0].append((node, node))
        for node in range(len(old_nodes[0]), node_indices):
            edges[1].append((node, node))

    return nodes, edges

def build_meta_relation(relations, nb_users):
    meta_relations = []
    for relation in relations:
        meta_relations.append(np.matmul(relation[:nb_users], relation[:nb_users].T))
    return meta_relations

def create_sparsemat_from_edgelist(edgelist, m, n):
    if not isinstance(edgelist, np.ndarray):
        edgelist = np.array(edgelist)
    rows, cols = edgelist[:,0], edgelist[:,1]
    ones = np.ones(len(rows), np.uint8)
    mat = sparse.coo_matrix((ones, (rows, cols)), shape=(m, n))
    return mat.tocsr()

def create_adjmat_from_edgelist(edgelist, size):
    adjmat = [[0]*size for _ in range(size)]
    for from_, to_ in edgelist:
        adjmat[from_][to_] = 1
    return np.array(adjmat, dtype=np.uint8)

def extend_featspace(feats: List[np.ndarray]):
    """
    Func: [Nu*fu,Nt*ft] -> (Nu+Nt)*(fu+ft)
    Solu: concat each feat-space, fill other positions with zero
    """
    full_dim  = sum([feat.shape[1] for feat in feats], 0)
    front_dim = 0
    extend_feats = []
    for feat in feats:
        nb_node = feat.shape[0]
        extend_feat = np.concatenate(
            (np.zeros(shape=(nb_node,front_dim)), feat, np.zeros(shape=(nb_node,full_dim-front_dim-feat.shape[1])))
        , axis=1)
        extend_feats.append(extend_feat)
        front_dim += feat.shape[1]
    return np.vstack(extend_feats)

def extend_wholegraph(g, ut_mp, initial_feats, tweet_per_user=20, sparse_graph=True):
    """
    功能: 在subg_deg483子图和ut_mp的基础上, 根据tweet_per_user参数重新生成hadjs和feats
    参数: sparse_graph=False时, 返回的实际上是一同质图
    """
    user_nodes = g.vs["label"]
    tweet_nodes, _, ut_edges = sample_tweets_around_user(users=set(user_nodes), ut_mp=ut_mp, tweets_per_user=tweet_per_user, return_edges=True)
    tweet_nodes = list(tweet_nodes)
    logger.info(f"nb_users={len(user_nodes)}, nb_tweets={len(tweet_nodes)}")

    # Users: 44896, Tweets: 10008103, Total: 10052999
    nodes, edges = reindex_graph([user_nodes, tweet_nodes], [g.get_edgelist(), ut_edges])

    if sparse_graph:
        uu_mat = create_sparsemat_from_edgelist(edges[0], len(nodes), len(nodes))
        ut_mat = create_sparsemat_from_edgelist(edges[1], len(nodes), len(nodes))
        hadjs = [uu_mat, ut_mat]
    else:
        hadjs = build_meta_relation([create_adjmat_from_edgelist(edges[0], len(nodes)), create_adjmat_from_edgelist(edges[1], len(nodes))], nb_users=len(user_nodes))

    user_feats, tweet_feats = initial_feats
    if len(user_feats) != len(user_nodes):
        user_feats = user_feats[user_nodes]
    tweet_feats = tweet_feats[tweet_nodes]

    if sparse_graph:
        feats = np.concatenate((
            np.append(user_feats, np.zeros(shape=(user_feats.shape[0], tweet_feats.shape[1])),  axis=1), 
            np.append(np.zeros(shape=(tweet_feats.shape[0], user_feats.shape[1])), tweet_feats, axis=1), 
        ), axis=0)
    else:
        feats = user_feats
    return hadjs, feats

def unique_cascades(df):
    unique_df = {}
    for hashtag, cascades in df.items():
        unique_cs, us = [], set()
        for user, timestamp in cascades:
            if user in us:
                continue
            us.add(user)
            unique_cs.append((user,timestamp))
        unique_df[hashtag] = unique_cs
    return unique_df

def gen_pos_neg_users(g, cascades, sample_ratio, stage):
    pos_users, neg_users = set(), set()
    all_activers = set([elem[0] for elem in cascades])
    max_ts = cascades[0][1] + int((cascades[-1][1]-cascades[0][1]+Ntimestage-1)/Ntimestage) * (stage+1)
    for user, ts in cascades:
        if ts > max_ts:
            break
        # Add Pos Sample
        pos_users.add(user)

        # Choos Neg from Neighborhood
        first_order_neighbor = list(set(g.neighborhood(user, order=1)) - all_activers)
        if len(first_order_neighbor) > 0:
            neg_user = random.choices(first_order_neighbor, k=min(len(first_order_neighbor), sample_ratio))
            neg_users |= set(neg_user)
        else:
            second_order_neighbor = list(set(g.neighborhood(user, order=2)) - all_activers)
            if len(second_order_neighbor) > 0:
                neg_user = random.choices(second_order_neighbor, k=min(len(first_order_neighbor), sample_ratio))
                neg_users |= set(neg_user)
    # logger.info(f"pos={len(pos_users)}, neg={len(neg_users)}, diff={len(pos_users & neg_users)}")
    return pos_users, neg_users

def gen_pos_neg_users2(g, cascades, sample_ratio):
    pos_users, neg_users = set(), set()
    all_activers = set([elem[0] for elem in cascades])
    for user, _ in cascades:
        # Add Pos Sample
        pos_users.add(user)

        # Choos Neg from Neighborhood
        first_order_neighbor = list(set(g.neighborhood(user, order=1)) - all_activers)
        if len(first_order_neighbor) > 0:
            neg_user = random.choices(first_order_neighbor, k=min(len(first_order_neighbor), sample_ratio))
            neg_users |= set(neg_user)
        else:
            second_order_neighbor = list(set(g.neighborhood(user, order=2)) - all_activers)
            if len(second_order_neighbor) > 0:
                neg_user = random.choices(second_order_neighbor, k=min(len(first_order_neighbor), sample_ratio))
                neg_users |= set(neg_user)
    # logger.info(f"pos={len(pos_users)}, neg={len(neg_users)}, diff={len(pos_users & neg_users)}")
    return pos_users, neg_users

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_labels(args, g, cascades):
    pos_users, neg_users = gen_pos_neg_users(g=g, cascades=cascades, sample_ratio=args.sample_ratio, stage=args.stage)
    num_user = g.vcount()

    # NOTE: label=1表示正样本, label=-1表示负样本, label=0表示未被选择
    labels = torch.zeros(num_user)
    labels[list(pos_users)] =  1
    labels[list(neg_users)] = -1

    float_mask = np.zeros(len(labels))
    for label in [-1,1]:
        ids = np.where(labels == label)[0]
        float_mask[ids] = np.random.permutation(np.linspace(1e-10,1,len(ids))) if args.shuffle else np.linspace(1e-10,1,len(ids))
    
    train_ids = np.where((float_mask>0) & (float_mask<=args.train_ratio/100))[0]
    val_ids   = np.where((float_mask>args.train_ratio/100) & (float_mask<=(args.train_ratio+args.valid_ratio)/100))[0]
    test_ids  = np.where(float_mask>(args.train_ratio+args.valid_ratio)/100)[0]
    # logger.info(f"train/valid/test={len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    train_mask = get_binary_mask(num_user, train_ids)
    val_mask   = get_binary_mask(num_user, val_ids)
    test_mask  = get_binary_mask(num_user, test_ids)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    
    pos_neg_labels = list(filter(lambda x: x==1 or x==-1, labels))
    nb_classes = np.unique(pos_neg_labels).shape[0]
    class_weight = torch.FloatTensor(len(pos_neg_labels) / (nb_classes*np.unique(pos_neg_labels, return_counts=True)[1])) if args.class_weight_balanced else torch.ones(nb_classes)
    labels[labels==-1] = 0
    
    return labels.long(), train_mask, val_mask, test_mask, nb_classes, class_weight

def get_centroids(X, y):
    nb_kind = len(np.unique(y))
    centers_ = []
    for idx in range(nb_kind):
        center = np.median(X[y==idx],axis=0)
        centers_.append([elem for elem in center])
    return np.array(centers_)

def get_centroids2(X, y):
    clf = NearestCentroid()
    clf.fit(X, y)
    centroids_ = clf.centroids_
    return np.array([[elem for elem in center] for center in centroids_])

def apply_clustering_algo(tweet_features, model:str='agg', desc='', thr=None, todraw=False):
    """
    NOTE: desc的一般形式为f"g{g.vcount()}", 而当model='agg'时形式为f"g{g.vcount()}_thr{thr}"
    """
    full_desc = f"{desc}"
    if model == 'agg':
        if thr is None:
            thr = 0.5 # default
        full_desc += f"_thr{thr}"
    centroids_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/clustering/{full_desc}_{model}_centroids_.pkl")
    labels_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/clustering/{full_desc}_{model}_labels_.pkl")
    if os.path.exists(centroids_filepath) and os.path.exists(labels_filepath):
        return load_pickle(centroids_filepath), load_pickle(labels_filepath)

    if not isinstance(tweet_features, np.ndarray):
        tweet_features = np.array(tweet_features)
    
    if todraw:
        # Reduce Dimensions for plotting
        pca_ = PCA(n_components=3)
        tweet_features = pca_.fit_transform(tweet_features)
    
    # 1. sample a fraction of tweets for clustering and training cluster-classifier
    frac = 1e-3
    train_tweets = random.choices(tweet_features, k=int(len(tweet_features)*frac))
    train_tweets_df = pd.DataFrame(train_tweets)
    logger.info("--Finish Sampling Tweets--")

    # 3. perform clustering
    if model == 'agg':
        logger.info(f"--Perform Hierarchical Clustering Method (thr={thr})--")
        model_ = AgglomerativeClustering(n_clusters=None, distance_threshold=thr) # thr is set to <default> 
        labels_ = model_.fit_predict(train_tweets_df).astype("int")
        # 3.2 train classifier and give labels to other test data
        knnmodel_ = KNeighborsClassifier(n_neighbors=1)
        knnmodel_.fit(train_tweets_df, labels_)
        whole_labels_ = knnmodel_.predict(tweet_features)
        logger.info(f"--Finish Training KNNModel and Calculating {model.capitalize()} Labels--")
        # 3.1 get centroids
        centroids = get_centroids2(X=tweet_features, y=whole_labels_)
        logger.info(f"--Finish Calculating {model.capitalize()} Centroids (Num={model_.n_clusters_}, Shape={centroids.shape})--")
    elif model == 'mbk':
        # 2. determine the K
        elbow_ = KElbowVisualizer(MiniBatchKMeans(), k=(4,30))
        elbow_.fit(train_tweets_df)
        n_clusters = elbow_.elbow_value_
        logger.info(f"--Finish Determining K={n_clusters}--")

        model_ = MiniBatchKMeans(n_clusters=n_clusters, random_state=2023)
        # 3.2 train classifier and give labels to all data
        whole_labels_ = model_.fit_predict(tweet_features).astype("int")
        # 3.1 get centroids
        centroids = model_.cluster_centers_
        logger.info(f"--Finish Calculating {model.capitalize()} Labels and Centroids--")
    save_pickle(centroids, centroids_filepath)
    save_pickle(whole_labels_, labels_filepath)

    # 4. plot
    if todraw:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection="3d", label="bla", computed_zorder=False)
        cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
        ax.scatter(train_tweets_df[0], train_tweets_df[1], train_tweets_df[2], c=labels_, marker=".", cmap=cmap, zorder=0)
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2], c='red', marker="o", s=100)
        plt.show()
    
    return centroids, whole_labels_

def get_tweet_feat_for_tweet_nodes(lda_model_k=20):
    processed_texts = load_pickle(f"")
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(processed_texts)
    LDA_model = LatentDirichletAllocation(n_components=lda_model_k, max_iter=50, random_state=2023)
    LDA_model.fit(dtm)
    doc_topic = LDA_model.transform(dtm)

    # save_pickle(cv,  f"cv/cv_0{texts_idx}_k{args.k}_maxiter50.p")
    # save_pickle(dtm, f"dtm/dtm_0{texts_idx}_k{args.k}_maxiter50.p")
    # save_pickle(doc_topic, f"doc2topic/doc_topic_0{texts_idx}_k{args.k}_maxiter50.p")
    # save_pickle(LDA_model, f"model/model_0{texts_idx}_k{args.k}_maxiter50.p")

    panel = pyLDAvis.sklearn.prepare(LDA_model, dtm, cv, mds='tsne') # Create the panel for the visualization
    pyLDAvis.save_html(panel, 'LDA-vis.html') # TODO: filename

def get_tweet_feat_for_user_nodes(lda_model_k=25):
    twft_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/lda-model/twft_per_user_k{lda_model_k}.pkl")
    if os.path.exists(twft_filepath):
        logger.info(f"Trying to Get Twft in path:{twft_filepath}...Success")
        return load_pickle(twft_filepath)
    
    twft = []
    prefix, suffix = os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model"), f"_k{lda_model_k}_maxiter{50}"
    logger.info(f"Calculating Twft Using CountVectorizer/LDAModel/ProcessedUserTexts in path:{prefix}/.../{suffix}")
    for part in range(1,11):
        cv_ = load_pickle(f"{prefix}/cv/cv_0{part}{suffix}.p")
        lda_model_ = load_pickle(f"{prefix}/model/model_0{part}{suffix}.p")
        user_texts_l = load_pickle(f"{prefix}/processedtexts-per-user/ProcessedTexts_{part}.p")

        twft.extend(lda_model_.transform(cv_.transform(user_texts_l)))
    save_pickle(twft, twft_filepath)
    return twft

def tweet_centralized_process(homo_g, user_tweet_mp, tweet_features, clustering_algo, distance_threshold):
    # 1. prepare tweet feature
    lda_k = tweet_features.shape[1]

    # 2. apply clustering algo
    centroids_, tw2centroids_ = apply_clustering_algo(tweet_features, model=clustering_algo, desc=f"g{homo_g.vcount()}", thr=distance_threshold)

    # replace user_tweet_edges with user_tweet_centroid_edges
    user_nodes = homo_g.vs["label"]
    user_tweet_centroid_edges = []
    for user in user_nodes:
        for tweet in user_tweet_mp[user]:
            user_tweet_centroid_edges.append((user, tw2centroids_[tweet]))

    # 3. build new homo graph
    tweet_nodes = [elem for elem in range(len(centroids_))]
    uu_edges = homo_g.get_edgelist()
    logger.info(f"Graph Info: nb-users={len(user_nodes)}, nb-tweets={len(tweet_nodes)}, nb-uu-edges={len(uu_edges)}, nb-ut-edges={len(user_tweet_centroid_edges)}")
    nodes, edges = reindex_graph([user_nodes, tweet_nodes], [uu_edges, user_tweet_centroid_edges])

    # 4. get node features for both users and tweets
    twft_for_users  = get_tweet_feat_for_user_nodes(lda_model_k=lda_k)
    feats = []
    feats.extend(list(np.array(twft_for_users)[user_nodes]))
    feats.extend(centroids_) # twft_for_tweets
    logger.info(f"New Homo Graph Info: nb-nodes={len(nodes)}, nb-edges={len(edges[0])}:{len(edges[1])}, nb-tweets={len(feats)}*{len(feats[0])}")

    return nodes, edges, np.array(feats)
