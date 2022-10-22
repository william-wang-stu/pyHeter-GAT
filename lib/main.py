import datetime
import logging
import pandas as pd
import os
from typing import Dict, Any, Tuple, List
from scipy import sparse
import numpy as np
import torch
import math
import torch.optim as optim
from itertools import chain
from model import HeterogeneousGraphAttention
from loss import TypeAwareAlignmentLoss
from torch import nn
from torch import Tensor
from scipy.sparse import csr_matrix
import shutil
from tensorboard_logger import tensorboard_logger
import argparse

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter

# Load Original Dataset

dir_filepath = "/root/Heter-GAT/data/twitter-foursquare anonymized"

df_fsquare_follows = pd.read_pickle(os.path.join(dir_filepath, "fsquare-follows.pkl.gz"))
df_fsquare_locations = pd.read_pickle(os.path.join(dir_filepath, "fsquare-locations.pkl.gz"))
df_fsquare_tips = pd.read_pickle(os.path.join(dir_filepath, "fsquare-tips.pkl.gz"))
df_fsquare_users = pd.read_pickle(os.path.join(dir_filepath, "fsquare-users.pkl.gz"))

df_twitter_follows = pd.read_pickle(os.path.join(dir_filepath, "twitter-follows.pkl.gz"))
df_twitter_tweets = pd.read_pickle(os.path.join(dir_filepath, "twitter-tweets.pkl.gz"))
df_twitter_users = pd.read_pickle(os.path.join(dir_filepath, "twitter-users.pkl.gz"))

node_types = ["User", "Tweet", "Location"]
edge_types = ["U-U", "U-T", "U-L"]
network_types = ["Twitter", "Foursquare"]

df_mp = {
    "Twitter": {
        "User": (df_twitter_users, "username"),
        "Tweet": (df_twitter_tweets, "text"),
        "Location": (df_twitter_tweets, "lat", "lng"),

        "U-U": (df_twitter_follows, "user1", "user2"),
        "U-T": (df_twitter_tweets, "username", "text"),
        "U-L": (df_twitter_tweets, "username", "lat", "lng"),
    }, 
    "Foursquare": {
        "User": (df_fsquare_users, "id"),
        "Tweet": (df_fsquare_tips, "text"),
        "Location": (df_fsquare_locations, "id"),

        "U-U": (df_fsquare_follows, "user1", "user2"),
        "U-T": (df_fsquare_tips, "user_id", "text"),
        "U-L": (df_fsquare_tips, "user_id", "loc_id"),
    },
}

nodes = {}

# Init Var: edges
for network_type in network_types:
    nodes[network_type] = {}
    for node_type in node_types:
        nodes[network_type][node_type] = {}
        with open(f"../output/graph/nodemap-{network_type}-{node_type}.txt", "r") as f:
            for idx, line in enumerate(f):
                if network_type == "Twitter" and node_type == "Tweet" and idx > 50000:
                    break
                if network_type == "Twitter" and node_type == "Location" and idx > 100000:
                    break
                parts = line[:-1].split(' ')
                # NOTE: strange!!! Tweet Nodes Nums are different between r and w
                if len(parts) < 2:
                    continue
                key, value = parts
                if key.isdigit():
                    key = int(key)
                nodes[network_type][node_type][key] = int(value)

edges = {}

def get_node_types(edge_type: str) -> Tuple[str, str]:
    from_type, to_type = edge_type.split("-")
    from_type = list(filter(lambda x: x[0] == from_type, node_types))[0]
    to_type = list(filter(lambda x: x[0] == to_type, node_types))[0]
    return from_type, to_type

# Init Var: edges
for network_type in network_types:
    edges[network_type] = {}
    for edge_type in edge_types:
        edges[network_type][edge_type] = []
        with open(f"../output/graph/edgelist-{network_type}-{edge_type}.txt", "r") as f:
            for line in f:
                from_, to_ = line[:-1].split(' ')
                if network_type == "Twitter" and edge_type == "U-T" and (int(from_) > 50000 or int(to_) > 50000):
                    continue
                if network_type == "Twitter" and edge_type == "U-L" and (int(from_) > 100000 or int(to_) > 100000):
                    continue
                edges[network_type][edge_type].append((from_, to_))

"""
Twitter Nodes: num_users=5223, num_tweets=6960800, num_locs=256497
Twitter Edges: num_U-U=164919, num_U-T=8178952, num_U-L=514610
Foursquare Nodes: num_users=5392, num_tweets=46617, num_locs=94187
Foursquare Edges: num_U-U=76972, num_U-T=48585, num_U-L=48585

Twitter Nodes: num_users=5223, num_tweets=6960800, num_locs=256497
Twitter Nodes: num_users=5223, num_tweets=6960377, num_locs=256497
Twitter Edges: num_U-U=164919, num_U-T=8178952, num_U-L=514610
Foursquare Nodes: num_users=5392, num_tweets=46617, num_locs=94187
Foursquare Nodes: num_users=5392, num_tweets=46616, num_locs=94187
Foursquare Edges: num_U-U=76972, num_U-T=48585, num_U-L=48585
"""
for network_type in network_types:
    num_users  = len(nodes[network_type]["User"])
    num_tweets = len(nodes[network_type]["Tweet"])
    num_locs   = len(nodes[network_type]["Location"])
    logger.info(f"{network_type} Nodes: total={num_users+num_tweets+num_locs}, num_users={num_users}, num_tweets={num_tweets}, num_locs={num_locs}")

    # num_users  = len(new_nodes[network_type]["User"])
    # num_tweets = len(new_nodes[network_type]["Tweet"])
    # num_locs   = len(new_nodes[network_type]["Location"])
    # logger.info(f"{network_type} Nodes: num_users={num_users}, num_tweets={num_tweets}, num_locs={num_locs}")

    num_uu = len(edges[network_type]["U-U"])
    num_ut = len(edges[network_type]["U-T"])
    num_ul = len(edges[network_type]["U-L"])
    logger.info(f"{network_type} Edges: total={num_uu+num_ut+num_ul}, num_U-U={num_uu}, num_U-T={num_ut}, num_U-L={num_ul}")

# NOTE: 所需构造的异构图是(|Rs|, N, N), N=N_user+N_tweet+N_location
# -> 送入第一层次的GAT时, 不同异构图的邻接矩阵不同, 得到的N*D'的emb也不同, 且每次只从中选取(N_user,D')的emb以备后用
# -> 送入第二层次的GAT时, 我们默认N=N_user

matrices = {}

def create_sparse(coo_list, m, n):
    data = np.ones((len(coo_list),))
    row = [pair[0] for pair in coo_list]
    col = [pair[1] for pair in coo_list]
    matrix = sparse.coo_matrix((data, (row, col)), shape=(m, n))
    return matrix

def extend_edges(edges: List[Any], num1: int, num2: int):
    extended_edges = []
    for (from_, to_) in edges:
        extended_edges.append((int(from_)+num1, int(to_)+num2))
    return extended_edges

for network_type in network_types:
    matrices[network_type] = {}

    start_idx_mp = {}
    indices = 0
    for node_type in node_types:
        start_idx_mp[node_type] = indices
        indices += len(nodes[network_type][node_type])
    # logger.info(start_idx_mp)

    for edge_type in edge_types:
        node1_t, node2_t = get_node_types(edge_type)
        extended_edges = edges[network_type][edge_type]
        if start_idx_mp[node1_t] or start_idx_mp[node2_t]:
            logger.info(f"node1_t={node1_t}, node2_t={node2_t}, start_idx_node1_t={start_idx_mp[node1_t]}, start_idx_node2_t={start_idx_mp[node2_t]}")
            extended_edges = extend_edges(extended_edges, start_idx_mp[node1_t], start_idx_mp[node2_t])
        matrices[network_type][edge_type] = create_sparse(extended_edges, indices, indices)

def get_gpu_mem_utils():
    logger.info("Allocated: ", torch.cuda.memory_allocated(0)/1024**3)
    logger.info("Cached: ", torch.cuda.memory_reserved(0)/1024**3)

def get_sparse_tensor(mat: sparse.coo_matrix):
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

initial_features = {}

for network_type in network_types:
    indices = 0
    for node_type in node_types:
        indices += len(nodes[network_type][node_type])
    initial_features[network_type] = torch.rand(indices, 100)
    if torch.cuda.is_available():
        initial_features[network_type] = initial_features[network_type].cuda()

hadj = {}
for network_type in network_types:
    hadj[network_type] = []
    for edge_type in edge_types:
        adj = get_sparse_tensor(matrices[network_type][edge_type])
        if torch.cuda.is_available():
            adj = adj.cuda()
        hadj[network_type].append(adj)

# Anchor Link Matrix
def get_anchor_link_matrix():
    anchor_links = []
    df = df_fsquare_users[["id", "twitter"]].dropna()
    for _, row in df.iterrows():
        if row["id"] in nodes["Foursquare"]["User"] and row["twitter"] in nodes["Twitter"]["User"]:
            anchor_links.append((nodes["Twitter"]["User"][row["twitter"]], nodes["Foursquare"]["User"][row["id"]]))
    # logger.info(f"Total Anchor Links={len(anchor_links)}")
    return create_sparse(anchor_links, len(nodes["Twitter"]["User"]), len(nodes["Foursquare"]["User"]))

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-log', type=str, default='', help="name of this run")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--k', type=int, default=30, help="Candidate Number in Evaluating Metrics P@k and MAP@k")
    parser.add_argument('--check-point', type=int, default=10, help="Check point")
    parser.add_argument('--train-ratio', type=float, default=80, help="Training ratio (0, 100)")
    parser.add_argument('--valid-ratio', type=float, default=0, help="Validation ratio (0, 100)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def init_random(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

def init_tensorboard(args):
    tensorboard_log_dir = 'tensorboard/HGAT_%s' % (args.tensorboard_log)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    shutil.rmtree(tensorboard_log_dir)
    tensorboard_logger.configure(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

args = init_args()
init_random(args)
init_tensorboard(args)

def save(obj, filename):
    torch.save(obj, f"../output/{filename}")

def load(filename):
    return torch.load(f"../output/{filename}")

# Train, Test, Eval
#   model() -> loss, eval/metrics(Precision@k, MAP@k) -> log
# Metrics: 参考Paper "DeepLink_A_Deep_Learning_Approach_for_User_Identity_Linkage"的相关定义
#   Precision@k: 考虑每个用户的前K个用户中是否存在真实的Pair-Label, 存在置1否则置0, 结果为所有用户该值的平均
#   MAP@k: 考虑每个用户的真实Pair-Label的排名Ranki, 结果为\Sigma{1/Ranki}/N

def get_train_test_pairs(ratio: float=0.8):
    """
    Args:
        ratio: train数据所占比例; 暂时只划分train/test数据集(不包括valid数据集)
    Return:
        train_row, train_col, test_row, test_col
    """
    M = get_anchor_link_matrix()
    length = len(M.row)
    counts = math.ceil(length * ratio)
    indices = np.random.choice(length, counts, replace=False)
    other_indices = list(set(range(length)) - set(indices))

    return M.row[indices], M.col[indices], M.row[other_indices], M.col[other_indices]

train_row, train_col, test_row, test_col = get_train_test_pairs(ratio=args.train_ratio/100)

# Evaluation

def cosine(xs: torch.Tensor, ys: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Efficiently calculate the pairwise cosine similairties between two set of vectors.

    Args:
        xs: feature matrix, [N, dim]
        ys: feature matrix, [M, dim]
        epsilon: a small number to avoid dividing by zero

    Retrun:
        a [N, M] matrix of pairwise cosine similairties
    """
    mat = xs @ ys.t()
    x_norm = xs.norm(2, dim=1) + epsilon
    y_norm = ys.norm(2, dim=1) + epsilon
    x_diag = (1 / x_norm).diag()
    y_diag = (1 / y_norm).diag()
    return x_diag @ mat @ y_diag

def calc_metrics(sims: torch.Tensor, k: int, rows: List[int], cols: List[int]) -> tuple:
    """
    Calculate the Precision@k and MAP@k from two sides, i.e., source-to-target and target-to-source.

    Args:
        sims: similarity matrix
        k: number of candidates
        rows, cols: index pairs of matched users, i.e., the ground truth

    Return:
        Precision@k, MAP@k
    """
    target = sims[rows, cols].reshape((-1, 1))
    left = sims[rows]
    right = sims.t()[cols]
    # match users from source to target
    preck_l, mapk_l = score(left, target, k)
    # match users from target to source
    preck_r, mapk_r = score(right, target, k)
    # averaging the scores from both sides
    return (preck_l + preck_r) / 2, (mapk_l + mapk_r) / 2

def score(mat: torch.Tensor, target: torch.Tensor, k: int) -> tuple:
    """
    Precision@k: 考虑每个用户的前K个用户中是否存在真实的Pair-Label, 存在置1否则置0, 结果为所有用户该值的平均
    MAP@k: 考虑每个用户的真实Pair-Label的排名Ranki, 结果为\Sigma{1/Ranki}/N
    """
    # number of users with similarities larger than the matched users
    rank = (mat >= target).sum(1)
    # rank = min(rank, k + 1)
    rank = rank.min(torch.tensor(k + 1))
    # precision@k
    preck = (rank < k+1).float().mean()
    # map@k
    mapk = (1./rank).mean()
    return preck.tolist(), mapk.tolist()

# Training and Testing

# 1. Prepare Data
alm = get_anchor_link_matrix()

# 2. Model, Optimizer, Loss
model_twitter = HeterogeneousGraphAttention(n_user=len(nodes["Twitter"]["User"]), n_units=[100, 64])
model_fsquare = HeterogeneousGraphAttention(n_user=len(nodes["Foursquare"]["User"]), n_units=[100, 64])
if torch.cuda.is_available():
    model_twitter.cuda()
    model_fsquare.cuda()

optimizer = optim.Adam(chain(model_twitter.parameters(), model_fsquare.parameters()), lr=1e-3, weight_decay=5e-4)

loss_fn = TypeAwareAlignmentLoss(epsilon=0)
# loss_fn = nn.L1Loss()

# Training and Testing

def eval(epoch_idx, args, log_desc="valid", data={
    "model": (model_twitter, model_fsquare),
    "loss": loss_fn,
    "train_pairs": (train_row, train_col),
    "test_pairs" : (test_row,  test_col),
}):
    # 1.
    ratio = args.train_ratio
    k = args.k
    alm = get_anchor_link_matrix()
    (model_twitter, model_fsquare) = data["model"]
    loss_fn = data["loss"]
    train_row, train_col = data["train_pairs"]
    test_row, test_col   = data["test_pairs"]

    # 2.
    model_twitter.eval()
    model_fsquare.eval()

    output_tw = model_twitter(initial_features["Twitter"], hadj["Twitter"])
    output_fs = model_fsquare(initial_features["Foursquare"], hadj["Foursquare"])
    loss_val = loss_fn(alm, output_tw, output_fs)

    # 3. Metrics
    # sims = get_loss_matrix(output_l[0], output_l[1])
    # NOTE: use type-fusion-emb to create sims matrix
    sims = cosine(output_tw[:,0].cpu(), output_fs[:,0].cpu())
    preck, mapk = calc_metrics(sims, k, train_row, train_col)
    logger.info(f"{log_desc} in Epoch={epoch_idx}: Loss={loss_val}, Precision@{k}={preck}, MAP@{k}={mapk} Using Training Pairs")
    save(sims, f"test/training_sims_epoch_{epoch_idx}_ratio_{ratio}_k{k}")

    # mask similarities of matched user pairs in the training set
    sims[train_row] = 0
    sims[:,train_col] = 0
    preck, mapk = calc_metrics(sims, k, test_row, test_col)
    logger.info(f"{log_desc} in Epoch={epoch_idx}: Loss={loss_val}, Precision@{k}={preck}, MAP@{k}={mapk} Using Testing Pairs")
    save(sims, f"test/testing_sims_epoch_{epoch_idx}_ratio_{ratio}_k{k}")

    # 4. Save
    save(output_tw, f"test/{log_desc}_emb_Twitter_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(output_fs, f"test/{log_desc}_emb_Foursquare_epoch_{epoch_idx}_ratio_{ratio}_k{k}")

def train(epoch_idx, args, log_desc="train", data={
    "model": (model_twitter, model_fsquare),
    "optimizer": optimizer,
    "loss": loss_fn,
}):
    # 1. Prepare Data
    alm = get_anchor_link_matrix()
        
    # 2. Model, Optimizer, Loss
    (model_twitter, model_fsquare), optimizer, loss_fn = data["model"], data["optimizer"], data["loss"]

    # 3. 
    model_twitter.train()
    model_fsquare.train()

    optimizer.zero_grad()
    output_tw = model_twitter(initial_features["Twitter"], hadj["Twitter"])
    output_fs = model_fsquare(initial_features["Foursquare"], hadj["Foursquare"])

    loss_train = loss_fn(alm, output_tw, output_fs)
    loss_train.backward()
    optimizer.step()

    logger.info(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3}")
    logger.info(f"Cached: {torch.cuda.memory_reserved(0)/1024**3}")
    logger.info(f"{log_desc} in Epoch={epoch_idx}: Loss={loss_train:f}")
    tensorboard_logger.log_value('train_loss', loss_train, epoch_idx+1)

    if (epoch_idx+1) % args.check_point == 0:
        logger.info(f"Checkpoint in Epoch={epoch_idx}")
        eval(epoch_idx, args)
    
    # 4. Save Model, Output Results and Loss
    ratio = args.train_ratio
    k = args.k
    save(model_twitter, f"train/model_twitter_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(model_fsquare, f"train/model_fsquare_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(output_tw, f"train/{log_desc}_emb_Twitter_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(output_fs, f"train/{log_desc}_emb_Fsquare_epoch_{epoch_idx}_ratio_{ratio}_k{k}")

def data_parallel_loss(output_tw: torch.Tensor, output_fs: torch.Tensor, alm=alm, loss_fn=loss_fn, parallel_num: int=7):
    # Split ALM and Output Values to Calculate Loss on Two GPU Devices
    #   思路: 只划分alm.row和output_tw, 即每次只考虑output_tw[part]和output_fs的loss

    step = int((alm.shape[0]+parallel_num-1)/parallel_num)
    # print(step)

    alm_l, output_tw_l = [], []
    alm_indices = 0

    for sidx in range(parallel_num):
        start_idx = sidx*step
        end_idx   = (sidx+1)*step if sidx < parallel_num-1 else alm.shape[0]
        # print(f"start_idx={start_idx}, end_idx={end_idx}")

        indices_sub = []
        for row, col in zip(alm.row[alm_indices:], alm.col[alm_indices:]):
            if row >= end_idx:
                break
            indices_sub.append((row-start_idx, col))
            alm_indices += 1
        alm_sub = create_sparse(indices_sub, step, alm.shape[1])
        alm_l.append(alm_sub)

        output_tw_l.append(output_tw[start_idx:end_idx])
    
    # Calculate Loss and Backward
    device_l = [3,4,5,6,7,8,9]
    loss_train = 0

    for sidx in range(parallel_num):
        device = f'cuda:{device_l[sidx]}'
        alm_sub = alm_l[sidx]
        output_tw_sub = output_tw_l[sidx].to(device)
        output_fs_sub = output_fs.to(device)
        loss_train_sub = loss_fn(alm_sub, output_tw_sub, output_fs_sub)
        loss_train += loss_train_sub.to('cuda:0')

    return loss_train

def train_parallel(epoch_idx, args, log_desc="train", data={
    "model": (model_twitter, model_fsquare),
    "optimizer": optimizer,
    "loss": loss_fn,
}):
    # 1. Prepare Data
    alm = get_anchor_link_matrix()
        
    # 2. Model, Optimizer, Loss
    (model_twitter, model_fsquare), optimizer, loss_fn = data["model"], data["optimizer"], data["loss"]

    # 3. 
    model_twitter.train()
    model_fsquare.train()

    optimizer.zero_grad()

    x1 = initial_features["Twitter"].to('cuda:1')
    hadj1 = [elem.to('cuda:1') for elem in hadj["Twitter"]]
    model_twitter = model_twitter.to('cuda:1')
    output_tw = model_twitter(x1, hadj1)

    x2 = initial_features["Foursquare"].to('cuda:2')
    hadj2 = [elem.to('cuda:2') for elem in hadj["Foursquare"]]
    model_fsquare = model_fsquare.to('cuda:2')
    output_fs = model_fsquare(x2, hadj2)

    loss_train = data_parallel_loss(output_tw, output_fs, alm, loss_fn)
    loss_train.backward()
    optimizer.step()

    for i in range(torch.cuda.device_count()):
        logger.info(f"Device idx={i} Allocated: {torch.cuda.memory_allocated(i)/1024**3}")
        logger.info(f"Device idx={i} Cached: {torch.cuda.memory_reserved(i)/1024**3}")
    logger.info(f"{log_desc} in Epoch={epoch_idx}: Loss={loss_train:f}")
    tensorboard_logger.log_value('train_loss', loss_train, epoch_idx+1)

    if (epoch_idx+1) % args.check_point == 0:
        logger.info(f"Checkpoint in Epoch={epoch_idx}")
        eval(epoch_idx, args)
    
    # 4. Save Model, Output Results and Loss
    ratio = args.train_ratio
    k = args.k
    save(model_twitter, f"train/model_twitter_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(model_fsquare, f"train/model_fsquare_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(output_tw, f"train/{log_desc}_emb_Twitter_epoch_{epoch_idx}_ratio_{ratio}_k{k}")
    save(output_fs, f"train/{log_desc}_emb_Fsquare_epoch_{epoch_idx}_ratio_{ratio}_k{k}")

def main():
    # Training
    logger.info("Start Training")
    for epoch_idx in range(args.epochs):
        train(epoch_idx, args)
    logger.info("End Training")

    # Testing
    logger.info("Start Testing")
    eval(args.epochs, args, "test")
    logger.info("End Testing")

    # Save
    save(model_twitter, f"train/model_twitter_epoch_{epoch_idx}_ratio_{args.train_ratio}_k{args.k}")
    save(model_fsquare, f"train/model_fsquare_epoch_{epoch_idx}_ratio_{args.train_ratio}_k{args.k}")

# main()
# train(0, args)
# eval(10, args)
train_parallel(0, args)
logger.info("Done")
