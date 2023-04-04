"""
总体思路: 
1. 输入数据: 异质图网络, 异质图节点特征
-   使用子网训练测试, 每个子网Sample上仅包括User节点, 这里为子图中的每个节点随机选取若干(N=20)个Tweet节点一并用作测试
    P.S. 注意此时子网大小为50Users+50*30Tweets=1550
2. 网路
3. 训练测试

新:
1. 以用户为中心的图注意力网络
2. 以推文为中心的图注意力网络
"""
# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import get_sparse_tensor
from lib.log import logger
from utils.utils import *
from utils.graph import *
from utils.tweet_clustering import tweet_centralized_process
from src.model import DenseGAT, HeterSparseGAT, HyperGAT, HyperGATWithHeterSparseGAT
import numpy as np
import argparse
import shutil
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

def load_labels(args, g, cascades):
    """
    功能: 生成训练测试用的正负样本标签
    """
    # 1. 根据传播级联cascades生成正负样本, 且生成负样本时会做正负样本均衡
    # pos_users, neg_users = gen_pos_neg_users(g=g, cascades=cascades, sample_ratio=args.sample_ratio, stage=args.stage)
    max_ts = cascades[0][1] + int((cascades[-1][1]-cascades[0][1]+Ntimestage-1)/Ntimestage) * (args.stage+1)
    pos_users, neg_users = gen_pos_neg_users2(g=g, cascades=[elem for elem in cascades if elem[1]<=max_ts], sample_ratio=args.sample_ratio)

    # 2. 划分训练测试验证集
    num_user = g.vcount()
    labels = torch.zeros(num_user)
    float_mask = np.zeros(num_user)

    labels[list(pos_users)] =  1
    labels[list(neg_users)] = -1
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
    
    # 3. 计算NLL Loss中所需的class_weight
    pos_neg_labels = list(filter(lambda x: x==1 or x==-1, labels))
    nb_classes = np.unique(pos_neg_labels).shape[0]
    class_weight = torch.FloatTensor(len(pos_neg_labels) / (nb_classes*np.unique(pos_neg_labels, return_counts=True)[1])) if args.class_weight_balanced else torch.ones(nb_classes)
    labels[labels==-1] = 0 # 置负样本标签为0而非-1
    
    return labels.long(), train_mask, val_mask, test_mask, nb_classes, class_weight

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='hetersparsegat', help="available options are ['densegat','hetersparsegat','hypergat','hypergatwithhetersparsegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
# >> Preprocess
parser.add_argument('--sample-ratio', type=int, default=1, help="Sampling Ratio (1~inf)")
parser.add_argument('--tweet-per-user', type=int, default=10, help="Tweets Per User (20, 40, 100)")
# parser.add_argument('--selected-tags', type=str, default="all", help="Agg On Some Part of Hastags")
parser.add_argument('--cluster-method', type=str, default="agg", help="Cluster Method, options are ['mbk', 'agg']")
parser.add_argument('--agg-dist-thr', type=float, default=5.0, help="Distance Threshold used in Hierarchical Clustering Method")
parser.add_argument('--embedding-method', type=str, default="bertopic", help="Embedding Method For Tweet Nodes")
# >> Model
parser.add_argument('--instance-normalization', action='store_true', default=False, help="Enable instance normalization")
parser.add_argument('--sparse-data', action='store_true', default=True, help="Use Sparse Data and Model (Only Valid When model='hypergat')")
parser.add_argument('--wo-user-centralized-net', action='store_true', default=False, help="Ablation Study w/o user-centralized-network")
parser.add_argument('--wo-tweet-centralized-net', action='store_true', default=False, help="Ablation Study w/o tweet-centralized-network")
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
# [default] hetersparsegat: 3e-3, hypergat: 3e-3, densegat: 3e-2
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--unified-dim', type=int, default=128, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--stage', type=int, default=Ntimestage-1, help="Time Stage (0~Ntimestage-1)")
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--train-ratio', type=float, default=60, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=20, help="Validation ratio (0, 100)")
parser.add_argument('--sota-test', action='store_true', default=False, help="Use Prepared Sota-Test-Dataset if set true")
parser.add_argument('--gpu', type=str, default="cuda:1", help="Select GPU")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Stage: Data Preparation
if not args.sota_test:
    g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_subgraph.p")) # Total 44896 User Nodes
    timelines = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/build_cascades/timeline_aggby_url_len5970_minuser5_subg.pkl"))
    # timelines = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/build_cascades/deg_le483_timeline_aggby_url_tag.pkl"))
else:
    # Sota-Test uses a down-sampled version of normal dataset, and corresponding down-sample-ratio is clarified in its filename
    g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subg_dp_20_100_ratio_35_20_2.p"))
    timelines = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subdf_dp_20_100_ratio_35_20_2.p"))
user_nodes = g.vs["label"]

structural_feat = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/vertex_feature_subgle483.npy"))
struc_cascade_feat = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/user_features_avg.p"))
deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deepwalk/deepwalk_added.emb_64"), max_idx=user_nodes[-1]+1)

fn = lambda val,ind: val if len(val)==len(ind) else val[ind]
structural_feat = fn(structural_feat, ind=user_nodes)
structural_feat = preprocessing.scale(structural_feat)
struc_cascade_feat = fn(struc_cascade_feat, ind=user_nodes)
struc_cascade_feat = preprocessing.scale(struc_cascade_feat)
deepwalk_feat = fn(deepwalk_feat, ind=user_nodes)
static_user_feat = np.concatenate((structural_feat,struc_cascade_feat),axis=1)

# TODO: consider args.stage when choosing tweet neighbors
# if isinstance(args.stage, int) and 0<=args.stage<=7:
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))
#     user_tweet_mp = user_tweet_mp[args.stage]
# else:
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))

if args.embedding_method == 'lda':
    user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))
    tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model/doc2topic_k25_maxiter50.p"))
elif args.embedding_method == 'bertopic':
    user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/ut_mp_filter_lt2words_processedforbert_subg.pkl"))
    tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/tweet-embedding/bertopic/topic_distribution_remove_hyphen_reduce_auto.pkl"))
if not isinstance(tweet_features, np.ndarray):
    tweet_features = np.array(tweet_features)

logger.info(f"user-feats dim={static_user_feat.shape[1]+deepwalk_feat.shape[1]}, tweet-feats dim={tweet_features.shape[1]}")

# tensorboard-logger Preparation
tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_t%d' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.tweet_per_user)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# Stage: Data Processing
timelines = unique_cascades(timelines) # remove duplicate appearances for single user in some particular hashtag
timelines = preprocess_timelines_byusernum(timelines, min_user_participate=20, max_user_participate=0)
logger.info(f"timeline number={len(timelines)}")

# labels_mp, train_mask_mp, val_mask_mp, test_mask_mp, class_weight_mp = {}, {}, {}, {}, {}
# for hashtag, cascades in timelines.items():
#     labels, train_mask, val_mask, test_mask, nb_classes, class_weight = load_labels(args=args, g=g, cascades=cascades)
#     labels_mp[hashtag] = labels; train_mask_mp[hashtag] = train_mask; val_mask_mp[hashtag] = val_mask; test_mask_mp[hashtag] = test_mask; class_weight_mp[hashtag] = class_weight
#     logger.info(f"hashtag={hashtag}, mask_len={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}, nb_classes={nb_classes}, class_weight={class_weight[0]:.2f}:{class_weight[1]:.2f}")

suffix = "_minuser10_onlyurl"
labels_mp = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/timeline_label_mp{suffix}.pkl"))
train_mask_mp = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/timeline_train_mask_mp{suffix}.pkl"))
val_mask_mp = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/timeline_val_mask_mp{suffix}.pkl"))
test_mask_mp = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/timeline_test_mask_mp{suffix}.pkl"))
class_weight_mp = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/timeline_class_weight_mp{suffix}.pkl"))

# Stage: Model Preparation
n_user = g.vcount()
n_units = [int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]
nb_classes = 2

if args.model == 'densegat':
    edges = g.get_edgelist() + [(i,i) for i in range(len(user_nodes))]
    adj = create_sparsemat_from_edgelist(edges, m=len(user_nodes), n=len(user_nodes))
    adj = get_sparse_tensor(adj.tocoo())

    static_emb = torch.FloatTensor(static_user_feat)
    dynamic_embs = [torch.FloatTensor(deepwalk_feat)]
    static_nfeat = static_emb.shape[1]
    dynamic_nfeats = [emb.shape[1] for emb in dynamic_embs]
    n_units = [int(x) for x in args.hidden_units.strip().split(",")] + [nb_classes]
    n_heads = n_heads + [1]
    model = DenseGAT(static_nfeat=static_nfeat, dynamic_nfeats=dynamic_nfeats, n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization,)
    logger.info(f"adj={adj.shape}, static_nfeats={static_nfeat}, dynamic_nfeats={dynamic_nfeats}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")
    
    if args.cuda:
        adj = adj.to(args.gpu)
        static_emb = static_emb.to(args.gpu)
        dynamic_embs = [emb.to(args.gpu) for emb in dynamic_embs]
        model.to(args.gpu)

elif args.model == 'hetersparsegat':
    tweet_nodes, _, ut_edges = sample_tweets_around_user(users=set(user_nodes), ut_mp=user_tweet_mp, tweets_per_user=args.tweet_per_user, return_edges=True)
    tweet_nodes = list(tweet_nodes)
    # nodes: Nu+Nt, edges: [(Nu+Nt)*(Nu+Nt),(Nu+Nt)*(Nu+Nt)]
    nodes, edges = reindex_graph(old_nodes=[user_nodes, tweet_nodes], old_edges=[g.get_edgelist(), ut_edges], add_self_loop=True)
    hadjs = [create_sparsemat_from_edgelist(edgelist=edges[0], m=len(nodes), n=len(nodes)),
        create_sparsemat_from_edgelist(edgelist=edges[1], m=len(nodes), n=len(nodes)),]
    hadjs = [get_sparse_tensor(hadj.tocoo()) for hadj in hadjs]
    
    static_user_features = np.concatenate((structural_feat,struc_cascade_feat),axis=1)
    fn = lambda val,ind: val if len(val)==len(ind) else val[ind]
    tweet_features = fn(tweet_features, tweet_nodes)
    feats = extend_featspace2([static_user_features, tweet_features])
    static_hembs  = [torch.FloatTensor(feats[0]), None]
    feats = extend_featspace2([deepwalk_feat, tweet_features])
    dynamic_hembs = [torch.FloatTensor(feat) for feat in feats]
    n_feats = [static.shape[1]+dynamic.shape[1] if static is not None else dynamic.shape[1] for static,dynamic in zip(static_hembs,dynamic_hembs)]
    dynamic_nfeats = [dynamic_emb.shape[1] for dynamic_emb in dynamic_hembs]
    model = HeterSparseGAT(n_feats=n_feats, dynamic_nfeats=dynamic_nfeats, n_unified=args.unified_dim, 
        n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, sparse=args.sparse_data)
    logger.info(f"hadjs={hadjs[0].shape}:{hadjs[1].shape}, static_hembs={static_hembs[0].shape}:{None}, dynamic_hembs={dynamic_hembs[0].shape}:{dynamic_hembs[1].shape}, n_user={n_user}, n_feats={n_feats}, dynamic_nfeats={dynamic_nfeats}, n_unified={args.unified_dim}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")

    if args.cuda:
        hadjs = [hadj.to(args.gpu) for hadj in hadjs]
        static_hembs  = [hemb.to(args.gpu) if hemb is not None else None for hemb in static_hembs]
        dynamic_hembs = [hemb.to(args.gpu) for hemb in dynamic_hembs]
        model.to(args.gpu)

elif args.model == 'hypergat':
    # tw_nodes: Nu+Nct, tw_edges: [(Nu+Nct)*(Nu+Nct),(Nu+Nct)*(Nu+Nct)], tw_feats: (Nu+Nct)*fct
    tw_nodes, tw_edges, tw_feats = tweet_centralized_process(homo_g=g, user_tweet_mp=user_tweet_mp, tweet_features=tweet_features, embedding_method=args.embedding_method, clustering_algo=args.cluster_method, distance_threshold=args.agg_dist_thr)
    user_edges = tw_edges[0]
    full_edges = sum([tw_edges[0], tw_edges[1]], [])
    hadjs = [create_sparsemat_from_edgelist(edgelist=user_edges, m=n_user, n=n_user) if args.sparse_data else 
            create_adjmat_from_edgelist(edgelist=user_edges, size=n_user), # Nu*Nu
        create_sparsemat_from_edgelist(edgelist=full_edges, m=len(tw_nodes), n=len(tw_nodes)) if args.sparse_data else
            create_adjmat_from_edgelist(edgelist=full_edges, size=len(tw_nodes)),] # (Nu+Nct)*(Nu+Nct)
    hadjs = [get_sparse_tensor(hadj.tocoo()) if args.sparse_data else torch.BoolTensor(hadj) for hadj in hadjs]

    fn = lambda val, ind: val if len(val)==len(ind) else val[ind]
    user_features = np.concatenate((structural_feat,struc_cascade_feat,deepwalk_feat),axis=1)
    user_features = fn(user_features, user_nodes)
    hembs = [torch.FloatTensor(user_features), torch.FloatTensor(tw_feats),]
    model = HyperGAT(n_feats=[user_features.shape[1], tw_feats.shape[1]], n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, sparse=args.sparse_data,
        wo_user_centralized_net=args.wo_user_centralized_net, wo_tweet_centralized_net=args.wo_tweet_centralized_net,)
    logger.info(f"hadjs={hadjs[0].shape}:{hadjs[1].shape}, hembs={hembs[0].shape}:{hembs[1].shape}, n_user={n_user}, n_feats={[user_features.shape[1],tweet_features.shape[1]]}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")

    if args.cuda:
        hadjs = [hadj.to(args.gpu) for hadj in hadjs]
        hembs = [hemb.to(args.gpu) for hemb in hembs]
        model.to(args.gpu)

elif args.model == 'hypergatwithhetersparsegat':
    # tw_nodes: Nu+Nct, tw_edges: [(Nu+Nct)*(Nu+Nct),(Nu+Nct)*(Nu+Nct)], tw_feats: (Nu+Nct)*fct
    tw_nodes, tw_edges, tw_feats = tweet_centralized_process(homo_g=g, user_tweet_mp=user_tweet_mp, tweet_features=tweet_features, clustering_algo=args.cluster_method, distance_threshold=args.agg_dist_thr)
    user_edges = tw_edges[0]
    full_edges = sum([tw_edges[0], tw_edges[1]], [])

    hadjs = [create_sparsemat_from_edgelist(edgelist=user_edges, m=n_user, n=n_user),                # (Nu,Nu)
        create_sparsemat_from_edgelist(edgelist=user_edges, m=len(tw_nodes), n=len(tw_nodes)),  # (Nu+Nct,Nu+Nct)
        create_sparsemat_from_edgelist(edgelist=full_edges, m=len(tw_nodes), n=len(tw_nodes)),] # (Nu+Nct,Nu+Nct)
    hadjs = [get_sparse_tensor(hadj.tocoo()) for hadj in hadjs]
    
    fn = lambda val, ind: val if len(val)==len(ind) else val[ind]
    user_features = np.concatenate((structural_feat,struc_cascade_feat,deepwalk_feat),axis=1)
    user_features  = fn(user_features, user_nodes)
    tweet_features = tw_feats[n_user:] # (Nu+Nct,fct) -> (Nct,fct)
    feats = extend_featspace([user_features, tweet_features]) # (Nu,fu), (Nct,fct) -> (Nu+Nct,fu+fct) 
    hembs = [torch.FloatTensor(user_features), torch.FloatTensor(feats)]
    model = HyperGATWithHeterSparseGAT(n_feats=[user_features.shape[1],tweet_features.shape[1]], n_unified=args.unified_dim, n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, sparse=args.sparse_data, wo_user_centralized_net=args.wo_user_centralized_net,)
    logger.info(f"hadjs={hadjs[0].shape}:{hadjs[1].shape}, hembs={hembs[0].shape}:{hembs[1].shape}, n_user={n_user}, n_feats={[user_features.shape[1],tweet_features.shape[1]]}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")

    if args.cuda:
        hadjs = [hadj.to(args.gpu) for hadj in hadjs]
        hembs = [hemb.to(args.gpu) for hemb in hembs]
        model.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# stopper = EarlyStopping(patience=args.patience)

def save_model(epoch, args, model, optimizer):
    state = {
        "epoch": epoch,
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/ckpt_epoch_{epoch}_model_{args.model}.pkl")
    torch.save(state, save_filepath)
    # help release GPU memory
    del state
    logger.info(f"Save State To Path={save_filepath}... Checkpoint Epoch={epoch}")

def load_model(filename, model):
    ckpt_state = torch.load(filename, map_location='cpu')
    model.load_state_dict(ckpt_state['model'])
    # optimizer.load_state_dict(ckpt_state['optimizer'])
    logger.info(f"Load State From Path={filename}... Checkpoint Epoch={ckpt_state['epoch']}")

def get_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-10)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    # logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
    return best_thr

def use_best_thr(best_thr, y_pred, y_score):
    # logger.info("using threshold %.4f", best_thr)
    y_score = np.array(y_score)
    y_pred = np.zeros_like(y_score)
    y_pred[y_score > best_thr] = 1

def evaluate(epoch, best_thrs=None, return_best_thr=False, log_desc='valid_'):
    model.eval()

    loss, correct, total, prec, rec, f1 = 0., 0., 0., 0., 0., 0.
    y_true, y_pred, y_score, thrs = [], [], [], {}
    for hashtag in sorted(timelines.keys()):
        labels, val_mask, test_mask, class_weight = labels_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag], class_weight_mp[hashtag]
        if args.cuda:
            labels, class_weight = labels.to(args.gpu), class_weight.to(args.gpu)
        
        if args.model == 'densegat':
            output = model(adj, static_emb, dynamic_embs)
        elif args.model == 'hetersparsegat':
            output = model(hadjs, static_hembs, dynamic_hembs)
        elif args.model == 'hypergat':
            output = model(hadjs, hembs)
        elif args.model == 'hypergatwithhetersparsegat':
            output = model([hadjs[0], [hadjs[1], hadjs[2]]], hembs)
        
        if best_thrs is None: # valid
            loss_batch = F.nll_loss(output[val_mask], labels[val_mask], class_weight)
            y_true_cur = labels[val_mask].data.tolist()
            y_pred_cur = output[val_mask].max(1)[1].data.tolist()
            y_true += y_true_cur
            y_pred += y_pred_cur
            y_score_cur = output[val_mask][:, 1].data.tolist()
            y_score += y_score_cur
            thrs[hashtag] = get_best_thr(y_true_cur, y_score_cur)
            alpha = val_mask.sum().item()
        else: # test
            loss_batch = F.nll_loss(output[test_mask], labels[test_mask], class_weight)
            y_true_cur = labels[test_mask].data.tolist()
            y_pred_cur = output[test_mask].max(1)[1].data.tolist()
            y_true += y_true_cur
            y_pred += y_pred_cur
            y_score_cur = output[test_mask][:, 1].data.tolist()
            y_score += y_score_cur
            use_best_thr(best_thrs[hashtag], y_pred_cur, y_score_cur)
            alpha = test_mask.sum().item()
        correct += alpha * np.mean(np.array(y_true_cur) == np.array(y_pred_cur))
        loss  += alpha * loss_batch.item()
        total += alpha

    model.train()

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f Acc: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f", log_desc, loss / total, correct / total, auc, prec, rec, f1)
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'acc', correct / total, epoch+1)
    writer.add_scalar(log_desc+'auc', auc, epoch+1)
    writer.add_scalar(log_desc+'prec', prec, epoch+1)
    writer.add_scalar(log_desc+'rec', rec, epoch+1)
    writer.add_scalar(log_desc+'f1', f1, epoch+1)

    if return_best_thr:
        return thrs
    else:
        return None

def train(epoch, log_desc='train_'):
    model.train()
    loss, total = 0., 0.
    for hashtag in sorted(timelines.keys()):
        labels, train_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], class_weight_mp[hashtag]
        if args.cuda:
            labels, class_weight = labels.to(args.gpu), class_weight.to(args.gpu)
        
        optimizer.zero_grad()
        if args.model == 'densegat':
            output = model(adj, static_emb, dynamic_embs)
        elif args.model == 'hetersparsegat':
            output = model(hadjs, static_hembs, dynamic_hembs)
        elif args.model == 'hypergat':
            output = model(hadjs, hembs)
        elif args.model == 'hypergatwithhetersparsegat':
            output = model([hadjs[0], [hadjs[1], hadjs[2]]], hembs)
        loss_train = F.nll_loss(output[train_mask], labels[train_mask], class_weight)
        alpha = train_mask.sum().item()
        loss  += alpha * loss_train.item()
        total += alpha
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)

    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thrs = evaluate(epoch, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, best_thrs=best_thrs, log_desc='test_')
        save_model(epoch, args, model, optimizer)

t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thrs = evaluate(args.epochs, return_best_thr=True, log_desc='valid_')
logger.info("testing...")
evaluate(args.epochs, best_thrs=best_thrs, log_desc='test_')
