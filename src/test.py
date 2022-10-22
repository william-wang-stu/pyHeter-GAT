# Adj -> Heter-Adj
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from lib.log import logger
from utils import load_pickle, save_pickle, summarize_distribution, find_rt_bound
import numpy as np

# NOTE: gen user feats
from utils import SubGraphSample

data_dirpath = "/root/data/HeterGAT/stages/stages_subg483_inf_40_1718027_deg_18_483_ego_20_neg_1_restart_20/"
samples = SubGraphSample(
    adj_matrices=np.load(os.path.join(data_dirpath, "adjacency_matrix.npy")),
    influence_features=np.load(os.path.join(data_dirpath, "influence_feature.npy")),
    vertex_ids=np.load(os.path.join(data_dirpath, "vertex_id.npy")),
    labels=np.load(os.path.join(data_dirpath, "label.npy")),
    tags=np.load(os.path.join(data_dirpath, "hashtag.npy")),
    time_stages=np.load(os.path.join(data_dirpath, "stage.npy"))
)
logger.info("Loading...")

inf_feats  = samples.influence_features
hashtags   = samples.tags
stages     = samples.time_stages

user_feats = []
for idx in range(len(samples)):
    if idx % 1000 == 0:
        logger.info(f"Progress={idx/len(samples)}")
    user_ids = samples.vertex_ids[idx]

    gra_feat = np.array(load_pickle(f"/root/data/HeterGAT/user_features/norm_gravity_feature/hashtag{hashtags[idx]}_t{stages[idx]}.p")).reshape(-1,1)
    exp_feat = np.array(load_pickle(f"/root/data/HeterGAT/user_features/norm_exptime_feature1/hashtag{hashtags[idx]}_t{stages[idx]}.p")).reshape(-1,1)
    cas_feat = np.array(load_pickle(f"/root/data/HeterGAT/user_features/norm_ce_feature/hashtag{hashtags[idx]}_t{stages[idx]}.p")).reshape(-1,1)
    user_feats.append(
        np.concatenate((inf_feats[idx], gra_feat[user_ids], exp_feat[user_ids], cas_feat[user_ids]), axis=1)
    )
user_feats = np.array(user_feats)
logger.info(f"{user_feats.shape}")
save_pickle(user_feats, os.path.join(data_dirpath, "user_features.p"))

# from typing import List
# import numpy as np
# from utils import SubGraphSample, HeterSubGraphSample
# import copy
# from scipy import sparse

# # from_dirname = "/root/data/HeterGAT/stages/stages_subg483_inf_40_1718027_deg_18_483_ego_50_neg_1_restart_20"
# from_dirname = "/root/data/HeterGAT/stages/stages_subg483_inf_40_1718027_deg_18_483_ego_20_neg_1_restart_20"
# # to_dirname   = "/root/data/HeterGAT/stages/hs_subg483_inf_40_1718027_deg_18_483_ego_50_neg_1_restart_20"
# to_dirname   = "/root/data/HeterGAT/stages/hs_subg483_inf_40_1718027_deg_18_483_ego_20_neg_1_restart_20"
# tweetid2userid_mp = load_pickle("/root/data/HeterGAT/basic/text/tweetid2userid_mp.p")
# tweet_feat = np.array(load_pickle("/root/data/HeterGAT/basic/doc2topic_tweetfeat.p"))
# tot_users_nb = 208894
# logger.info("Finish Loading Data...")

# def extend_adj_matrices(samples: SubGraphSample, tweet_ids: List[List[int]], heter_samples: HeterSubGraphSample):
#     adj_matrices = samples.adj_matrices
#     vertex_ids   = samples.vertex_ids

#     for idx in range(len(samples.labels)):
#         if idx and idx % 1000 == 0:
#             logger.info(f"idx={idx}")
#         extend_subnetwork = np.array(
#             np.concatenate((vertex_ids[idx], np.array(list(tweet_ids[idx])))), 
#             dtype=int
#         )
#         heter_samples.vertex_ids.append(extend_subnetwork) # (N,)
#         subnetwork_size, user_ids_nb = len(extend_subnetwork), len(vertex_ids[idx])
#         elem_idx_mp = {elem:idx for idx,elem in enumerate(extend_subnetwork)}

#         # Build U-U, U-T adj-matrices, respectively
#         heter_adj_matrices = []
#         uu_matrices = np.array([[0]*subnetwork_size for _ in range(subnetwork_size)], dtype=int)
#         uu_matrices[:user_ids_nb,:user_ids_nb] = adj_matrices[idx]
#         heter_adj_matrices.append(sparse.csr_matrix(uu_matrices)) # toarray()

#         # Get Corresponding User_id By Tweet_id, and then convert them into indexes in extend_subnetwork
#         ut_matrices = copy.deepcopy(uu_matrices)
#         for tweet_id in tweet_ids[idx]:
#             user_id = tweetid2userid_mp[tweet_id]
#             net_userid = elem_idx_mp[user_id]
#             net_tweetid = elem_idx_mp[tweet_id]
#             ut_matrices[net_userid][net_tweetid] = 1
#         heter_adj_matrices.append(sparse.csr_matrix(ut_matrices)) # toarray()
#         del uu_matrices, ut_matrices

#         heter_samples.heter_adj_matrices.append(heter_adj_matrices) # (|Rs|,N,N)
    
#     heter_samples.labels = samples.labels

# def extend_initial_features(samples: SubGraphSample, heter_samples: HeterSubGraphSample):
#     influence_features = samples.influence_features
#     stage = samples.time_stages[0]
#     hashtags = samples.tags

#     for idx in range(len(samples)):
#         if idx and idx % 1000 == 0:
#             logger.info(f"idx={idx}")
#         vertex_ids = heter_samples.vertex_ids[idx]
#         # Build User Features
#         # initial_features(user) = influence_features||Gra(hashtag,stage)||Exp(hashtag,stage)||Cas(hashtag,stage), dim=2
#         gra_feat = np.array(load_pickle(f"/root/data/HeterGAT/user_features/norm_gravity_feature/hashtag{hashtags[idx]}_t{stage}.p")).reshape(-1,1)
#         exp_feat = np.array(load_pickle(f"/root/data/HeterGAT/user_features/norm_exptime_feature1/hashtag{hashtags[idx]}_t{stage}.p")).reshape(-1,1)
#         cas_feat = np.array(load_pickle(f"/root/data/HeterGAT/user_features/norm_ce_feature/hashtag{hashtags[idx]}_t{stage}.p")).reshape(-1,1)
#         user_ids = samples.vertex_ids[idx]
#         user_features = np.concatenate((influence_features[idx], gra_feat[user_ids], exp_feat[user_ids], cas_feat[user_ids]), axis=1)

#         # Build Tweet Features
#         tweet_ids = vertex_ids[len(user_ids):]
#         tweet_features = tweet_feat[tweet_ids]

#         # Extend Two Features with Padding zeros
#         initial_features = np.concatenate((
#             np.append(user_features,  [[0]*tweet_features.shape[1] for _ in range(user_features.shape[0])],  1),
#             np.append([[0]*user_features.shape[1]  for _ in range(tweet_features.shape[0])], tweet_features, 1)
#         ), axis=0)
#         heter_samples.initial_features.append(initial_features)

# # Main Solution
# tweetids = load_pickle(f"{to_dirname}/tweet_ids.p")
# samples  = load_pickle(f"{to_dirname}/selected_samples.p")
# hetersamples = HeterSubGraphSample()
# extend_adj_matrices(samples, tweetids, hetersamples)
# extend_initial_features(samples, hetersamples)

# # Extend Vertex Indices
# for idx in range(len(hetersamples)):
#     vertex_ids = hetersamples.vertex_ids[idx]
#     hetersamples.vertex_ids[idx] = [index if index<tot_users_nb else index+tot_users_nb for index in vertex_ids]

# # 
# hetersamples.heter_adj_matrices = np.array(hetersamples.heter_adj_matrices)
# hetersamples.initial_features = np.array(hetersamples.initial_features)
# hetersamples.vertex_ids = np.array(hetersamples.vertex_ids)
# hetersamples.labels = np.array(hetersamples.labels)

# save_pickle(hetersamples, f"{to_dirname}/heter_samples.p")
# logger.info("Finish...")