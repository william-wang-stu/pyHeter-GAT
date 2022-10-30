import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from lib.log import logger
from lib.utils import get_node_types, extend_edges, get_sparse_tensor
import pickle
import numpy as np
from typing import Any, Dict, List
import igraph
import os
import random
import psutil
import argparse
import torch
import shutil
from tensorboard_logger import tensorboard_logger
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import copy
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
DATA_ROOTPATH = config['DEFAULT']['DataRootPath']

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
                logger.info(f"n={n}, d={d}")
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

def gen_random_tweet_ids(samples: SubGraphSample, outdir: str, tweets_per_user:int=5):
    tweet_ids = []
    sample_ids = []
    ut_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))

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
