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
import torch
from scipy import sparse
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
import bitermplus
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gsdmm import MovieGroupProcess
from itertools import combinations, chain
from numba import njit
from numba import types
from numba.typed import Dict, List
import pyLDAvis.sklearn

config = configparser.ConfigParser()
# NOTE: realpath(__file__)是在获取执行这段代码所属文件的绝对路径, 即~/pyHeter-GAT/src/utils.py
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini'))
DATA_ROOTPATH = config['DEFAULT']['DataRootPath']
Ntimestage = int(config['DEFAULT']['Ntimestage'])

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

def wrap(func):
    """
    Usage: wrap(graph.degree)(0)
    """
    def inner(*args, **kwargs):
        # logger.info(f"func={func}, args={args}, kwargs={kwargs}")
        return func(*args, mode="out", **kwargs)
    return inner

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

def sample_docs_foreachuser(docs, user_tweet_mp, sample_frac=0.01, min_sample_num=3):
    sample_docs = []
    for user_id in range(len(user_tweet_mp)):
        docs_id = user_tweet_mp[user_id]
        sample_num = int(sample_frac*len(docs_id))
        if sample_num < min_sample_num:
            continue
        user_docs = random.choices(docs[docs_id[0]:docs_id[-1]+1], k=sample_num)
        sample_docs.extend(user_docs)
    logger.info(f"sample_docs_num={len(sample_docs)}")
    return sample_docs

def lda_model(raw_texts, num_topics, visualize):
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(raw_texts)
    model = LatentDirichletAllocation(n_components=num_topics, max_iter=50, random_state=2023)
    model.fit(dtm)
    topic_distr = model.transform(dtm)
    if visualize:
        panel = pyLDAvis.sklearn.prepare(model, dtm, cv, mds='tsne') # Create the panel for the visualization
    else:
        panel = None
    return {
        "topic-distr": topic_distr,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": panel,
    }

def btmplus_model(raw_texts, num_topics, visualize):
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(raw_texts)
    vocab = np.array(cv.get_feature_names_out()) # get_feature_names_out is only available in ver1.0
    
    # replace words with its vocab word_ids
    docs_vec = bitermplus.get_vectorized_docs(raw_texts, vocab)
    biterms = bitermplus.get_biterms(docs_vec)
    model = bitermplus.BTM(dtm, vocab, T=num_topics)
    model.fit(biterms, iterations=30)

    topic_distr = model.transform(docs_vec)
    logger.info(f"METRICS: perplexity={model.perplexity_}, coherence={model.coherence_}") # model.labels_
    return {
        "topic-distr": topic_distr,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": None,
    }

def gsdmm_model(raw_texts, num_topics, visualize):
    # NOTE: we cant first sample train-corpus, bcz vocabs extracted from train-corpus is not enough for transforming the whole train+test corpus
    cv = CountVectorizer(stop_words="english")
    dtm = cv.fit_transform(raw_texts)
    n_docs, n_terms = dtm.shape
    model = MovieGroupProcess(K=num_topics, alpha=0.01, beta=0.01, n_iters=30)
    docs = [doc.split(' ') for doc in docs] # NOTE: Important!!!, since our sentence is string"Absolutely", if we dont split it, `for word in doc` would produce 'A b s...'
    # NOTE: around 1min/10,0000docs
    labels = model.fit(docs, n_terms)

    # get topic distribution
    # NOTE: we can rewrite mgp.py in gsdmm to save intermediate variables, i.e. m_z, n_z, n_z_w, d_z
    # model = MovieGroupProcess().from_data(K=20, alpha=0.01, beta=0.01, D=n_docs, vocab_size=n_terms, 
    #     cluster_doc_count=m_z, cluster_word_count=n_z, cluster_word_distribution=n_z_w)
    topic_distrs = []
    for doc in raw_texts: # NOTE: around 1min/10,0000docs
        topic_distrs.append(model.score(doc))
    return {
        "topic-distr": topic_distrs,
        "model": model,
        "cv": cv,
        "dtm": dtm,
        "pyvis-panel": None,
    }

def get_tweet_feat_for_tweet_nodes(model="lda", num_topics=20, visualize=False):
    """
    valid model names are ['lda','btm','gsdmm','bertopic']
    """
    suffix = f"model{model}" # used for naming saved-results, i.e. topic-distr_{suffix}.pkl
    if model == 'lda' or model == 'btm':
        suffix += f"_numtopic{num_topics}"

    sent_emb_dir = os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model/")
    raw_texts = load_pickle(os.path.join(sent_emb_dir, "processedtexts-per-text/ptexts_per_user_all.p"))
    if model == "lda":
        ret = lda_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    elif model == "btm":
        # ret = btm_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
        ret = btmplus_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    elif model == 'gsdmm':
        ret = gsdmm_model(raw_texts=raw_texts, num_topics=num_topics, visualize=visualize)
    elif model == 'bertopic':
        # NOTE: there are two ways for creating topic distributions, see details in reference https://github.com/MaartenGr/BERTopic/issues/1026
        # we choose to use func approximate_distribution(), since it is fasttttt!
        # Other options for accelaerating can be referenced in https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#gpu-acceleration
        model_filepath = os.path.join(sent_emb_dir, f"topic-distr_{suffix}.pkl")
        if os.path.exists(model_filepath):
            model = load_pickle(model_filepath)
            topic_distr, _ = model.approximate_distribution(raw_texts, min_similarity=1e-5)
            return
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        bertopic_model = BERTopic(language="english", nr_topics="auto", embedding_model=sentence_model)
        model = bertopic_model.fit(raw_texts)
        topic_distr, _ = model.approximate_distribution(raw_texts, min_similarity=1e-5)
        # topics = model._map_predictions(model.hdbscan_model.labels_)
        # probs = model.hdbscan_model.probabilities_
    
    # Save Final Results
    for key,val in ret.items():
        if val is None:
            continue
        save_pickle(val, f"{key}_{suffix}.pkl")

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
