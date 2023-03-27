from utils.utils import load_pickle, save_pickle
from lib.log import logger
import igraph
import os
import random
import re
import torch
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Any, Dict, List, Tuple
from itertools import combinations

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

def extend_featspace2(feats: List[np.ndarray]):
    """
    Func: [Nu*fu,Nt*ft] -> [(Nu+Nt)*fu,(Nu+Nt)*ft]
    """
    nb_nodes = sum([feat.shape[0] for feat in feats], 0)
    extend_feats = []
    front_dim = 0
    for feat in feats:
        extend_feat = np.concatenate(
            (np.zeros(shape=(front_dim, feat.shape[1])), feat, np.zeros(shape=(nb_nodes-front_dim-feat.shape[0], feat.shape[1])))
            , axis=0)
        extend_feats.append(extend_feat)
        front_dim += feat.shape[0]
    return extend_feats

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

def search_cascades(user_texts_mp:Dict[int,dict], subg:igraph.Graph)->Dict[int,list]:
    """
    功能: 将符合相同条件的推文组织为同一话题级联, 这些条件包括,
        1. 包含相同tag, i.e. #blacklivesmatter
        2. 包含相同url, i.e. 
        3. 用"RT @USER"形式标识的多个文本内容相同的推文
    """
    userid2graphid = {user_id:graph_id for graph_id, user_id in enumerate(subg.vs["label"])}
    timelines_tag = aggby_same_text(user_texts_mp, regex_pattern=r'(#\w+)', userid2graphid=userid2graphid)
    timelines_url = aggby_same_text(user_texts_mp, regex_pattern=r'(https?://[a-zA-Z0-9.?/&=:]+)', userid2graphid=userid2graphid)
    def preprocess(timelines:Dict[int,list], min_user_participate:int, max_user_participate:int)->Dict[int,list]:
        # Sort by Timestamp
        timelines = {key:sorted(value, key=lambda elem:elem[1]) for key,value in timelines.items()}
        # Remove Too Short or Too Long Cascades By Unique User-Nums
        user_participate_mp = {key:len(set([elem[0] for elem in value])) for key,value in timelines.items()}
        if max_user_participate < min_user_participate:
            max_user_participate = max([elem for elem in user_participate_mp.values()])
        timelines = {key:value for key,value in timelines.items() if user_participate_mp[key]>=min_user_participate and user_participate_mp[key]<=max_user_participate}
        return timelines
    timelines_tag = preprocess(timelines_tag, min_user_participate=5, max_user_participate=2000)
    timelines_url = preprocess(timelines_url, min_user_participate=5, max_user_participate=-1)
    return timelines_tag, timelines_url, None

def aggby_same_text(user_texts_mp:Dict[int,dict], regex_pattern:str, userid2graphid:Dict[int,int])->Dict[int,list]:
    """
    参数: 
        user_texts_mp: { user_id: { text_id: {'timestamp': , 'text': ,}, { text_id2: {} }, ... }, ... }
        regex_pattern: 过滤话题级联的正则表达式模式
            i.e. tag_pattern = r'(#\w+)', url_pattern = r'(https?://[a-zA-Z0-9.?/&=:]+)'
        userid2graphid: 原始拓扑图(v=208894)与采样子图(v=44896)的节点ID对应关系
            i.e. userid2graphid={user_id:graph_id for graph_id, user_id in enumerate(subg.vs["label"])}
    返回值:
        timelines: { tag: [(u1,t1),...], tag2:... }
    """
    valid_users = list(userid2graphid.keys())
    timelines:Dict[int,list] = {}

    for user_id, raw_texts in user_texts_mp.items():
        if user_id not in valid_users:
            continue
        for _, text in raw_texts.items():
            group = re.finditer(regex_pattern, text['text'])
            if group is None:
                continue
            for match in group:
                end_pos = match.span()[1]
                # incomplete tags shouldnt been recorded
                if end_pos == len(text['text'])-1 and text['text'][-1] == '…':
                    continue
                tag = match.group(1).lower()
                tag = tag.replace('…', '') # remove not filtered '…'
                if tag not in timelines:
                    timelines[tag] = []
                timelines[tag].append((userid2graphid[user_id], text['timestamp']))
    return timelines

def aggby_retweet_info(user_texts_mp:Dict[int,dict], userid2graphid:Dict[int,int]):
    """
    思路: 将去掉"RT @USER"形式后, 包含相同文本内容的推文组织成级联

    """
    # Aggregate RT Info
    retweet_pattern = r'RT @\w+: (.+)'
    timelines = aggby_same_text(user_texts_mp=user_texts_mp, regex_pattern=retweet_pattern, userid2graphid=userid2graphid)

    # Search for Possible Original Tweets
    valid_users = list(userid2graphid.keys())
    for retweet, _ in timelines.items():
        for user_id, raw_texts in user_texts_mp.items():
            if user_id not in valid_users:
                continue
            for _, text in raw_texts.items():
                if text['text'].lower() == retweet:
                    timelines[retweet].append((userid2graphid[user_id], text['timestamp']))

    return timelines

def find_tweet_by_cascade_info(user_texts_mp:Dict[int,dict], user_id:int, timestamp:int)->str:
    """
    Usage: find_tweet_by_cascade_info(user_texts_mp=user_texts, user_id=subg.vs[user]["label"], timestamp=ts)
    """
    texts = user_texts_mp[user_id]
    for _, text in texts.items():
        if text['timestamp'] == timestamp:
            return text['text']
    return None

def build_topic_similarity_user_edges(topic_labels:np.ndarray, ut_mp:dict):
    # 1. Get Doc-Topic
    # 2. Agg User-Topic
    # ut_mp = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/ut_mp_filter_lt2words_processedforbert_subg.pkl")
    user_ids = [[user]*len(tweet_ids) for user, tweet_ids in ut_mp.items()]
    user_ids = [item for sublist in user_ids for item in sublist]

    df = pd.DataFrame({
        'Topic_ID': topic_labels,
        'User_ID': user_ids,
    })
    # agg_topic_dict = df.groupby('User_ID').agg({'Topic_ID': set})['Topic_ID'].to_dict()
    agg_user_dict = df.groupby('Topic_ID').agg({'User_ID': set})['User_ID'].to_dict() # {topic_id: {user_id1,user_id2,...}}
    # df['Agg_Topic_ID'] = df['User_ID'].map(agg_topic_dict)

    # 3. Build Full-Connected User-User Edges
    user_edges_dict = {}
    for key, value in agg_user_dict.items():
        if key == -1: continue
        user_edges_dict[key] = list(combinations(value, r=2))
    
    return user_edges_dict
