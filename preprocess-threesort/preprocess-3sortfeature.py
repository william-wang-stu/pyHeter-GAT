import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from utils.util import *
import re

debug = True

# 1. Read Network and ActionLog
# graph = read_network()
# diffusion_dict = read_actionlog()
import pickle
def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

graph = load_pickle("/root/Lab_Related/data/Heter-GAT/Classic/graph/uugraph-directed.p")
diffusion_dict = load_pickle("/root/Lab_Related/data/Heter-GAT/Classic/ActionLog.p")

# 1-2. Calculate Common Vars
graph_vcount = graph.vcount()
Nslice = Ntimeslice + 1
active_users    = get_active_users(diffusion_dict=diffusion_dict)
candidate_users = set([i for i in range(graph_vcount)]) - active_users
active_users, candidate_users = list(active_users), list(candidate_users)

def compute_structural_features(graph):
    logger.info("Computing rarity (reciprocal of degree)")
    degree = np.array(graph.degree())
    degree[degree==0] = 1
    rarity = 1. / degree
    logger.info("Computing clustering coefficient..")
    cc = graph.transitivity_local_undirected(mode="zero")
    logger.info("Computing pagerank...")
    pagerank = graph.pagerank(directed=False)
    logger.info("Computing authority_score...")
    authority_score = graph.authority_score()
    logger.info("Computing hub_score...")
    hub_score = graph.hub_score()
    logger.info("Computing evcent...")
    evcent = graph.evcent(directed=False)
    logger.info("Computing coreness...")
    coreness = graph.coreness()
    logger.info("Structural feature computation done!")
    structural_features = np.column_stack(
        (rarity, cc, pagerank,
            #constraint, closeness, betweenness,
            authority_score, hub_score, evcent, coreness)
    )
    
    os.makedirs(outputfile_dirpath, exist_ok=True)
    with open(os.path.join(outputfile_dirpath, "vertex_feature.npy"), "wb") as f:
        np.save(f, structural_features)

# compute_structural_features(graph=graph)

# 2. Triple-Feature
logger.info("Start to Calculate Triple Features")

def no_more_than_one_check(sample_l: list):
    for sample_elem in sample_l:
        assert sample_elem>=0 and sample_elem<=1

# 2-1. Calculate Gravity Feature: 1 * |V-Candidates|
def get_gravity_feature(active_users, order_t, tao_t):
    gravity_feature = [0. for _ in range(graph_vcount)]
    # beta = 1.0 # no affect to ranking result, 因为大家都乘系数等于大家都不乘
    logger.info("Start Computing Gra-Feat")
    cnt = 0
    for active_user in active_users:
        cnt += 1
        if cnt % 1000 == 0:
            logger.info(f"[Debug] calculated {cnt} active users in Gravity Feature")

        activer_degree = graph.degree(active_user, mode="out")
        if activer_degree == 0:
            continue
        for order_len in range(1, 2+1):
            neighborhood = graph.neighborhood(active_user, order=order_len, mode="out", mindist=order_len)
            # logger.info(f"len={len(neighborhood)}")
            for neighbor_user in neighborhood:
                gravity_feature[neighbor_user] += activer_degree * graph.degree(neighbor_user, mode="in") / pow(order_len, 3)

    return gravity_feature

# for tau in [1,2,3]:
tau=3
gravity_feature = get_gravity_feature(
    active_users=active_users,
    order_t=2,
    tao_t=tau,
)

with open(os.path.join(outputfile_dirpath, f"gravity_feature_tau{tau}.npy"), "wb") as f:
    np.save(f, gravity_feature)
logger.info(f"Calculated Gravity Feature with hyper-param tao={tau}")

# 2-2. Exposure Time Feature: 1 * |V-Candidates|
def get_exposure_time_sum_distribution(diffusion_dict: Dict[str, Any], graph, BUCKET_NUM):
    exposure_time_sum = [0 for _ in range(graph_vcount)]

    if debug:
        nu = 0
    
    for _, cascade in diffusion_dict.items():
        if debug:
            nu += 1
            if nu % 10 == 0:
                logger.info(f"Calculated {nu} cascades in Exposure Time Sum Distribution")
    
        for user, timestamp in cascade[1:]:
            cascade_elem_idx = 0
            while cascade_elem_idx < len(cascade) and cascade[cascade_elem_idx][1] < timestamp:
                if user == cascade[cascade_elem_idx][0]:
                    cascade_elem_idx += 1
                    continue
                if cascade[cascade_elem_idx][0] in graph.neighbors(user):
                    exposure_time_sum[user] += timestamp - cascade[cascade_elem_idx][1]
                cascade_elem_idx += 1
        
    logger.info(f"exposure_time_sum elem>0 number: {len(list(filter(lambda item: item>0, exposure_time_sum)))}")

    if len(list(filter(lambda item: item>0, exposure_time_sum))) == 0:
        logger.info("Error: exposure_time_sum is straight zeros")
        return [], -1

    # split exposure_num into several buckets
    min_et_sum, max_et_sum = min(exposure_time_sum), max(exposure_time_sum)
    logger.info(f"min_exposure_num: {min_et_sum}, max_exposure_num: {max_et_sum}")

    bucket_range = (max_et_sum - min_et_sum) / BUCKET_NUM
    buckets = [0 for _ in range(BUCKET_NUM + 1)]
    for et_sum in exposure_time_sum:
        if et_sum > 0:
            bucket_idx = round((et_sum - min_et_sum) / bucket_range)
            buckets[bucket_idx] += 1
            # logger.info(bucket_idx, et_sum)
    # logger.info(f"buckets: {buckets}")

    return buckets, bucket_range

def get_timestamp_upper_bound(timestamp, time_lower_bound, time_upper_bound, Nslice):
    time_range = (time_upper_bound - time_lower_bound) / Nslice + 1e-4
    time_idx = int((timestamp - time_lower_bound) / time_range) + 1
    return min(time_upper_bound, time_idx * time_range + time_lower_bound)

def get_exposure_time_feature(et_buckets, bucket_range, graph, diffusion_dict: Dict[str, Any], BUCKET_NUM):
    exposure_time_feature = [0 for _ in range(graph_vcount)]
    et_buckets_sum = sum(et_buckets)

    if debug:
        nu = 0
    
    for _, cascade in diffusion_dict.items():
        if debug:
            nu += 1
            if nu % 1000 == 0:
                logger.info(f"[Debug] calculated {nu} cascades in Exposure Time Feature")
                
        lower_bound, upper_bound = cascade[0][1], cascade[len(cascade)-1][1]
        for active_user, timestamp in cascade:
            timestamp_upper_bound = get_timestamp_upper_bound(timestamp=timestamp, time_lower_bound=lower_bound, time_upper_bound=upper_bound, Nslice=Nslice)
            for target_user in graph.neighbors(active_user):
                exposure_time_feature[target_user] += timestamp_upper_bound - timestamp
    
    for etf_idx, etf_elem in enumerate(exposure_time_feature):
        if etf_elem > 0:
            bucket_idx = round( (etf_elem - 0) / bucket_range)
            if bucket_idx > BUCKET_NUM:
                bucket_idx = BUCKET_NUM
            exposure_time_feature[etf_idx] = et_buckets[bucket_idx] / et_buckets_sum

    # for active_user in active_users:
    #     exposure_time_feature[active_user] = 1
    
    logger.info(f"exposure_time_feature non-zero distribution: {np.unique(list(filter(lambda item:item>0, exposure_time_feature)), return_counts=True)}")

    return exposure_time_feature

# for BUCKET_NUM in [1000, 10000, 100000]:
BUCKET_NUM = 10000
buckets, bucket_range = get_exposure_time_sum_distribution(
    diffusion_dict=diffusion_dict,
    graph=graph,
    BUCKET_NUM=BUCKET_NUM
)
exposure_time_feature = get_exposure_time_feature(
    et_buckets=buckets,
    bucket_range=bucket_range,
    graph=graph,
    diffusion_dict=diffusion_dict,
    BUCKET_NUM=BUCKET_NUM
)

with open(os.path.join(outputfile_dirpath, f"exposure_time_feature_net{BUCKET_NUM}.npy"), "wb") as f:
    np.save(f, exposure_time_feature)
logger.info(f"Calculated Exposure Time Feature with hyper-param Net={BUCKET_NUM}")

# 2-3. Cascade Embedding Feature: 1 * |V-Candidates|
# Generate CE-Vector with Skip-Gram, Results are in cascade_embedding sub-directory

ce_vec_dict = {}
vocab_size, embed_size = 0, 0
# with open(cascade_embedding_matrix_filepath, 'r') as f:
with open("/root/Lab_Related/data/Heter-GAT/Classic/deepwalk_emb64.embeddings", 'r') as f:
    for idx, line in enumerate(f):
        if idx == 0:
            vocab_size, embed_size = line.split(' ')
        else:
            idx, emb = line.split(' ', 1)
            # emb = re.sub('\s|\t|\n|"|\'', '', emb[1:-2])
            # emb = [float(elem) for elem in emb.split(',')]
            emb = [float(elem) for elem in emb.split(' ')]
            ce_vec_dict[int(idx)] = emb
logger.info("Read Cascade Embedding Matrix")

# Normalize Each Vector
norm_ce_vec_dict = {}
for key, value in ce_vec_dict.items():
    norm_ce_vec_dict[key] = value / np.linalg.norm(value)

def get_cascade_embedding_feature(active_users, graph_vcount):
    cascade_embedding_feature = [0 for _ in range(graph_vcount)]

    ce_matrix = np.array([norm_ce_vec_dict[active_user] for active_user in active_users]).reshape(-1,128)
    active_user_num = ce_matrix.shape[0]

    for user in range(graph_vcount):
        intermediate_result = np.matmul(ce_matrix, norm_ce_vec_dict[user])
        cascade_embedding_feature[user] = active_user_num - intermediate_result.sum()

    # cascade_embedding_feature = stable_sigmoid(cascade_embedding_feature)
    # for active_user in active_users:
    #     cascade_embedding_feature[active_user] = 1
    
    return cascade_embedding_feature

cascade_embedding_feature = get_cascade_embedding_feature(
    active_users=active_users,
    graph_vcount=graph_vcount
)

with open(os.path.join(outputfile_dirpath, "feature/cascade_embedding_feature.npy"), "wb") as f:
    np.save(f, cascade_embedding_feature)
logger.info("Calculated Cascade Embedding Feature")
