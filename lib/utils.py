import torch
import math
import pandas as pd
from scipy import sparse
import numpy as np
import os
from typing import Dict, Any, Tuple, List
from .log import logger

# Constant

node_types = ["User", "Tweet", "Location"]
edge_types = ["U-U", "U-T", "U-L"]
network_types = ["Twitter", "Foursquare"]

# dir_filepath = "/root/Heter-GAT/lib/data/twitter-foursquare anonymized"

# df_fsquare_follows = pd.read_pickle(os.path.join(dir_filepath, "fsquare-follows.pkl.gz"))
# df_fsquare_locations = pd.read_pickle(os.path.join(dir_filepath, "fsquare-locations.pkl.gz"))
# df_fsquare_tips = pd.read_pickle(os.path.join(dir_filepath, "fsquare-tips.pkl.gz"))
# df_fsquare_users = pd.read_pickle(os.path.join(dir_filepath, "fsquare-users.pkl.gz"))

# df_twitter_follows = pd.read_pickle(os.path.join(dir_filepath, "twitter-follows.pkl.gz"))
# df_twitter_tweets = pd.read_pickle(os.path.join(dir_filepath, "twitter-tweets.pkl.gz"))
# df_twitter_users = pd.read_pickle(os.path.join(dir_filepath, "twitter-users.pkl.gz"))

### 原始数据格式分析
#   Initial Feature List (Randomly Initialized)
#       TODO: user profile可以作为User的Initial Feature的若干维度, i.e. Location
#   Heterogeneous Adjacency Matrix <- Load Data
#   Anchor Link Matrix <- Load Data

# Twitter <-> FourSquare
#   Anchor Link Matrix: fsquare-users.pkl.gz id<->twitter
#   NOTE: df["twitter"].dropna(), 有些FourSquare User不存在Twitter Id

# Foursquare Social Network
#   User: fsquare-users.pkl.gz id
#   Tweet: fsquare-tips.pkl.gz text <- 构造text_id
#   Loc: fsquare-locations.pkl.gz id <- 抽象为简单的数字loc_id
#   User-User: fsquare-follows.pkl.gz user1->user2
#   User-Tweet: fsquare-tips.pkl.gz user_id -> text <- text_id
#   User-Loc: fsquare-tips.pkl.gz user_id -> loc_id <- (int)loc_id

# Twitter Social Network
# NOTE: Loc类型的节点是没有的, 只有部分User(~514610/8178957)存在User-Loc(Lat,Lng)的关系
# 意思是只有少部分用户去了某些地方可能会附上定位坐标
# NOTE: 从数据分析来看(见该数据文件夹中的data-format.ipynb), 
# Location指的就是(Lat,Lng)组合, User-Location指的是user-(Lat,Lng)
#   User: twitter-users.pkl.gz username
#   Tweet: twitter-tweets.pkl.gz text <- 构造text_id
#   Loc: twitter-tweets.pkl.gz (lat,lng) <- 去NaN, 去重
#   User-User: twitter-follows.pkl.gz user1 -> user2
#   User-Tweet: twitter-tweets.pkl.gz username -> text <- text_id
#   User-Loc: twitter-tweets.pkl.gz username -> (lat,lng) <- 去NaN

# df_mp = {
#     "Twitter": {
#         "User": (df_twitter_users, "username"),
#         "Tweet": (df_twitter_tweets, "text"),
#         "Location": (df_twitter_tweets, "lat", "lng"),

#         "U-U": (df_twitter_follows, "user1", "user2"),
#         "U-T": (df_twitter_tweets, "username", "text"),
#         "U-L": (df_twitter_tweets, "username", "lat", "lng"),
#     }, 
#     "Foursquare": {
#         "User": (df_fsquare_users, "id"),
#         "Tweet": (df_fsquare_tips, "text"),
#         "Location": (df_fsquare_locations, "id"),

#         "U-U": (df_fsquare_follows, "user1", "user2"),
#         "U-T": (df_fsquare_tips, "user_id", "text"),
#         "U-L": (df_fsquare_tips, "user_id", "loc_id"),
#     },
# }

def get_gpu_mem_usage(device:int=0):
    logger.info("Allocated: ", torch.cuda.memory_allocated(device)/1024**3)
    logger.info("Cached: ", torch.cuda.memory_reserved(device)/1024**3)

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
    logger.info(f"preck_l={preck_l}, mapk_l={mapk_l}, preck_r={preck_r}, mapk_r={mapk_r}")
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

def save(obj, filename):
    torch.save(obj, f"../output/{filename}")

def load(filename):
    return torch.load(f"../output/{filename}")

# Renaming and Indexing All Nodes
# def set_indexer(nodes, network_type: str, node_type: str):
#     """
#     Usage:
#         # for network_type in network_types:
#         #     for node_type in node_types:
#         #         set_indexer(nodes, network_type, node_type)
#     """
#     df, *col_name = df_mp[network_type][node_type]
#     if len(col_name) == 1:
#         # df["username"].tolist()
#         elem_list = df[col_name[0]].unique()
#     else:
#         # df["lat"].dropna().astype(str) + df["lng"].dropna().astype(str)
#         elem_list = df[col_name].astype(str).agg('-'.join, axis=1).unique()
#     if node_type not in node_types:
#         logger.info("Error: NOT valid node_type")
#         return
#     mp = nodes[network_type][node_type]

#     indices = 0
#     for elem in elem_list:
#         mp[elem] = indices
#         indices += 1

def get_node_types(edge_type: str) -> Tuple[str, str]:
    from_type, to_type = edge_type.split("-")
    from_type = list(filter(lambda x: x[0] == from_type, node_types))[0]
    to_type = list(filter(lambda x: x[0] == to_type, node_types))[0]
    return from_type, to_type

def set_dtype(obj: Any, dtype: str) -> Any:
    if dtype == "object":
        return obj
    elif hasattr(obj, "astype") and type(obj) != dtype:
        return obj.astype(dtype)
    else:
        return obj

def filter_columns(df: pd.DataFrame, col_name: List[str], remove_nan: bool=True):
    if remove_nan:
        df = df[col_name][~df[col_name].isna().any(axis=1)]
    else:
        df = df[col_name]
    df.index = range(len(df))
    return df
    # if len(col_name) == 1:
    #     # df["username"].tolist()
    #     col_list = df[col_name[0]]
    # else:
    #     # df["lat"].dropna().astype(str) + df["lng"].dropna().astype(str)
    #     col_list = df[col_name].astype(str).agg('-'.join, axis=1).unique()
    # return col_list

# def add_edges(nodes, edges, network_type: str, edge_type: str, is_directed: bool=False):
#     """
#     # for network_type in network_types:
#     #     for edge_type in edge_types:
#     #         add_edges(nodes, edges, network_type, edge_type)
#     """
#     if edge_type not in edge_types:
#         logger.info("Error: NOT valid edge_type")
#         return
#     edge_mp = edges[network_type][edge_type]
#     from_type, to_type = get_node_types(edge_type)
#     from_mp = nodes[network_type][from_type]
#     to_mp   = nodes[network_type][to_type]

#     df, *col_name = df_mp[network_type][edge_type]
#     # logger.info(f"from_col={col_name[0:1]}, to_col={col_name[1:]}")
#     filtered_columns = filter_columns(df, col_name)
#     from_dtype, to_dtype = df.dtypes[col_name[0]], df.dtypes[col_name[1]]
#     # logger.info(from_dtype, to_dtype)
#     # logger.info(f"total rows={len(filtered_columns)}, type={type(filtered_columns)}")
#     for row_idx, row in filtered_columns.iterrows():
#         # if row_idx and row_idx % 100000 == 0:
#         #     logger.info(f"row_idx={row_idx}")
#         from_val, to_val = row[0], row[1:]
#         if len(row[1:]) > 1:
#             to_val = "-".join([str(elem) for elem in row[1:]])
#         else:
#             to_val = to_val[0]
#         from_val, to_val = set_dtype(from_val, from_dtype), set_dtype(to_val, to_dtype)
#         if from_val in from_mp and to_val in to_mp:
#             from_node, to_node = from_mp[from_val], to_mp[to_val]
#             edge_mp.append((from_node, to_node))
#     # logger.info(f"Total {len(edge_mp)} Edges, From_Node_Type={from_dtype}, To_Node_Type={to_dtype}")

def get_sparse_tensor(mat: sparse.coo_matrix):
    values = mat.data
    indices = np.vstack((mat.row, mat.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mat.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

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

def save_nodes(nodes):
    for network_type in network_types:
        for node_type in node_types:
            with open(f"../output/nodemap-{network_type}-{node_type}.txt", "w") as f:
                for key, value in nodes[network_type][node_type].items():
                    f.write(f"{str(key)} {int(value)}\n")

def load_nodes():
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
    
    return nodes

def save_edges(edges):
    for network_type in network_types:
        for edge_type in edge_types:
            with open(f"../output/edgelist-{network_type}-{edge_type}.txt", "w") as f:
                for (from_, to_) in edges[network_type][edge_type]:
                    f.write(f"{from_} {to_}\n")


def load_edges():
    edges = {}

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

    return edges

def build_adjacency_matrices(nodes, edges):
    """
    思路: 所需构造的异构图是(|Rs|, N, N), N=N_user+N_tweet+N_location
    -> 送入第一层次的GAT时, 不同异构图的邻接矩阵不同, 得到的N*D'的emb也不同, 且每次只从中选取(N_user,D')的emb以备后用
    -> 送入第二层次的GAT时, 我们默认N=N_user
    """
    matrices = {}

    for network_type in network_types:
        matrices[network_type] = {}

        start_idx_mp = {}
        indices = 0
        for node_type in node_types:
            start_idx_mp[node_type] = indices
            indices += len(nodes[network_type][node_type])

        for edge_type in edge_types:
            node1_t, node2_t = get_node_types(edge_type)
            extended_edges = edges[network_type][edge_type]
            if start_idx_mp[node1_t] or start_idx_mp[node2_t]:
                logger.info(f"node1_t={node1_t}, node2_t={node2_t}, start_idx_node1_t={start_idx_mp[node1_t]}, start_idx_node2_t={start_idx_mp[node2_t]}")
                extended_edges = extend_edges(extended_edges, start_idx_mp[node1_t], start_idx_mp[node2_t])
            matrices[network_type][edge_type] = create_sparse(extended_edges, indices, indices)

    return matrices

# def get_anchor_link_matrix(nodes):
#     anchor_links = []
#     df = df_fsquare_users[["id", "twitter"]].dropna()
#     for _, row in df.iterrows():
#         if row["id"] in nodes["Foursquare"]["User"] and row["twitter"] in nodes["Twitter"]["User"]:
#             anchor_links.append((nodes["Twitter"]["User"][row["twitter"]], nodes["Foursquare"]["User"][row["id"]]))
#     # logger.info(f"Total Anchor Links={len(anchor_links)}")
#     return create_sparse(anchor_links, len(nodes["Twitter"]["User"]), len(nodes["Foursquare"]["User"]))

# def get_train_test_pairs(nodes, ratio: float=0.8):
#     """
#     Args:
#         ratio: train数据所占比例; 暂时只划分train/test数据集(不包括valid数据集)
#     Return:
#         train_row, train_col, test_row, test_col
#     """
#     M = get_anchor_link_matrix(nodes)
#     length = len(M.row)
#     counts = math.ceil(length * ratio)
#     indices = np.random.choice(length, counts, replace=False)
#     other_indices = list(set(range(length)) - set(indices))

#     return M.row[indices], M.col[indices], M.row[other_indices], M.col[other_indices]
