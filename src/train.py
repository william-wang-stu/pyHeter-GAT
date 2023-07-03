# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import get_sparse_tensor
from lib.log import logger
from utils.utils import *
from utils.graph import *
from utils.tweet_clustering import tweet_centralized_process
from utils.metric import compute_metrics
from src.model import DenseSparseGAT, HeterEdgeSparseGAT
from src.sota.TAN.model import TAN
from src.sota.TAN.Option import Option
import numpy as np
import argparse
import shutil
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

DATA_ROOTPATH = "/".join(DATA_ROOTPATH.split('/')[:-1])
logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

def load_labels(g:igraph.Graph, cascades:List[Tuple[int,int]], args:dict)->Union[None,Tuple[torch.Tensor,torch.Tensor]]:
    """
    功能: 生成训练测试用的正负样本标签
    """

    # 1. 根据传播级联cascades生成正负样本, 且生成负样本时会做正负样本均衡
    # pos_users, neg_users = gen_pos_neg_users(g=g, cascades=cascades, sample_ratio=args.sample_ratio, stage=args.stage)
    max_ts = cascades[0][1] + int((cascades[-1][1]-cascades[0][1]+Ntimestage-1)/Ntimestage) * (args['stage']+1)
    pos_users, neg_users = gen_pos_neg_users2(g=g, cascades=[elem for elem in cascades if elem[1]<=max_ts], sample_ratio=args['sample_ratio'])

    # Direct Return for Unbalanced One-Class Label
    if len(pos_users) == 0 or len(neg_users) == 0:
        return None

    # 2. 生成labels和mask
    num_user = g.vcount()
    labels = torch.zeros(num_user)
    labels[list(pos_users)] = 1
    # labels[list(neg_users)] = 0
    labels = labels.long()

    mask = get_binary_mask(num_user, list(pos_users)+list(neg_users))
    if hasattr(torch, 'BoolTensor'):
        mask = mask.bool()
    
    # 3. 计算NLL Loss中所需的class_weight
    nb_classes = 2
    class_weight = torch.FloatTensor((len(pos_users)+len(neg_users))/(nb_classes*np.array([len(neg_users),len(pos_users)]))) if args['class_weight_balanced'] else torch.ones(nb_classes)
    
    return labels, mask, class_weight

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='densegat', help="available options are ['densegat','heteredgegat','hetersparsegat','hypergat','hypergatwithhetersparsegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
# >> Preprocess
parser.add_argument('--sample-ratio', type=int, default=1, help="Sampling Ratio (1~inf)")
parser.add_argument('--tweet-per-user', type=int, default=10, help="Tweets Per User (20, 40, 100)")
parser.add_argument('--min-user-participate', type=int, default=100, help="Min User Participate in One Cascade")
# parser.add_argument('--selected-tags', type=str, default="all", help="Agg On Some Part of Hastags")
parser.add_argument('--cluster-method', type=str, default="agg", help="Cluster Method, options are ['mbk', 'agg']")
parser.add_argument('--agg-dist-thr', type=float, default=5.0, help="Distance Threshold used in Hierarchical Clustering Method")
parser.add_argument('--embedding-method', type=str, default="llm", help="Embedding Method For Tweet Nodes")
# >> Model
parser.add_argument('--n-component', type=int, default=3, help="Number of Prominent Component Topic Classes Foreach Topic")
parser.add_argument('--window-size', type=int, default=200, help="Window Size of Building Topical Edges")
parser.add_argument('--instance-normalization', action='store_true', default=False, help="Enable instance normalization")
parser.add_argument('--sparse-data', action='store_true', default=True, help="Use Sparse Data and Model (Only Valid When model='hypergat')")
parser.add_argument('--use-tweet-feat', action='store_true', default=False, help="Use Tweet-Side Feat Aggregated From Tag Embeddings")
parser.add_argument('--wo-user-centralized-net', action='store_true', default=False, help="Ablation Study w/o user-centralized-network")
parser.add_argument('--wo-tweet-centralized-net', action='store_true', default=False, help="Ablation Study w/o tweet-centralized-network")
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
# [default] hetersparsegat: 3e-3, hypergat: 3e-3, densegat: 3e-2
parser.add_argument('--lr', type=float, default=3e-2, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--unified-dim', type=int, default=128, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--stage', type=int, default=Ntimestage-1, help="Time Stage (0~Ntimestage-1)")
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--check-point', type=int, default=1, help="Check point")
parser.add_argument('--train-ratio', type=float, default=0.8, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=0.1, help="Validation ratio (0, 100)")
parser.add_argument('--gpu', type=str, default="cuda:1", help="Select GPU")
# >> Ablation Study
parser.add_argument('--use-subgraph-dataset', action='store_true', default=False, help="Use Prepared Subgraph Dataset for DenseGAT-form Models if set true")
parser.add_argument('--use-url-timeline', action='store_true', default=False, help="Use URL-only Timeline Dataset if set true, otherwise Use Tag-only Timeline")
parser.add_argument('--use-random-multiedge', action='store_true', default=False, help="Use Random Multi-Edge to build Heter-Edge-Matrix if set true (Available only when model='heteredgegat')")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# $1 Stage: Data Preparation
# 1. Graph and Timeline
if not args.use_subgraph_dataset:
    g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_subgraph.p")) # Total 44896 User Nodes
    if not args.use_url_timeline: # Use TAG-only Dataset
        timelines = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/build_cascades/deg_le483_timeline_aggby_tag.pkl"))
    else: # Else Use URL-only Dataset
        timelines = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/build_cascades/timeline_aggby_url_len5970_minuser5_subg.pkl"))
else:
    # Sota-Test uses a down-sampled version of normal dataset, and corresponding down-sample-ratio is clarified in its filename
    g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subg_dp_20_100_ratio_35_20_2.p"))
    timelines = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subdf_dp_20_100_ratio_35_20_2.p"))
user_nodes = g.vs["label"]

# 2. User-Side Feats
structural_feat = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/deperacated/user_features/vertex_feature_subgle483.npy"))
three_sort_feat = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/deperacated/user_features/user_features_avg.p"))
deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deepwalk/deepwalk_added.emb_64"), max_idx=user_nodes[-1]+1)

# 3. User-Tweet-Mp, Tweet-Side Feats
if args.embedding_method == 'lda':
    # user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))
    tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/deperacated/lda-model/doc2topic_k25_maxiter50.p"))
elif args.embedding_method == 'bertopic':
    # user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp_filter_lt2words_processedforbert_subg.pkl"))
    tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/deperacated/tweet-embedding/bertopic/topic_distribution_remove_hyphen_reduce_auto.pkl"))
elif args.embedding_method == 'llm':
    use_url_suffix = "_timeline_url" if args.use_url_timeline else ""
    tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/deperacated/tweet-embedding/llm/tag_embs_aggbyuser{use_url_suffix}_model_xlm-roberta-base_pca_dim{args.unified_dim}.pkl"))
if not isinstance(tweet_features, np.ndarray):
    tweet_features = np.array(tweet_features)

# TODO: consider args.stage when choosing tweet neighbors
# if isinstance(args.stage, int) and 0<=args.stage<=7:
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))
#     user_tweet_mp = user_tweet_mp[args.stage]
# else:
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))

# $2 Stage: Preparation
# 1. tensorboard-logger Preparation
tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_t%d' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.tweet_per_user)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# 2. Preprocess Timeline
preprocess_timelines = unique_cascades(timelines) # remove duplicate appearances for single user in some particular hashtag
preprocess_timelines = {key:value for key,value in preprocess_timelines.items() if not key[1:].isdigit()}
preprocess_timelines = preprocess_timelines_byusernum(preprocess_timelines, min_user_participate=args.min_user_participate, max_user_participate=0)
logger.info(f"timelines={len(timelines)}, preprocess timelines={len(preprocess_timelines)}, total caselems={sum([len(elem) for elem in preprocess_timelines.values()])}")

# 3. Preprocess User-Side & Tweet-Side Feats
fn = lambda val,ind: val if len(val)==len(ind) else val[ind]
def preprocess_static_feat(static_feat:np.ndarray)->np.ndarray:
    static_feat = fn(static_feat, ind=user_nodes)
    static_feat = preprocessing.scale(static_feat)
    return static_feat

structural_feat = preprocess_static_feat(structural_feat)
three_sort_feat = preprocess_static_feat(three_sort_feat)
deepwalk_feat = fn(deepwalk_feat, ind=user_nodes)
# tweet_features = reduce_dimension(tweet_features, args.unified_dim)

# 4. Build tag2label_mask_cw_mp
use_url_suffix = "_url5970" if args.use_url_timeline else ""
tag2label_mask_cw_mp_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/build_tag2label_mask_cw_mp/deg_le483_tag2label_mask_cw_mp{use_url_suffix}.pkl")
if os.path.exists(tag2label_mask_cw_mp_filepath):
    obj = load_pickle(tag2label_mask_cw_mp_filepath)
    labels_mp = obj['label']; train_mask_mp = obj['train_mask']; val_mask_mp = obj['val_mask']; test_mask_mp = obj['test_mask']; class_weight_mp = obj['class_weight']
else:
    labels_mp, train_mask_mp, val_mask_mp, test_mask_mp, class_weight_mp = {}, {}, {}, {}, {}
    for hashtag, cascades in preprocess_timelines.items():
        ret = load_labels(g=g, cascades=cascades, args=vars(args))
        if ret is None:
            logger.info(f"hashtag={hashtag}, one class not existing...")
            continue
        labels, train_mask, val_mask, test_mask, class_weight = ret
        labels_mp[hashtag] = labels; train_mask_mp[hashtag] = train_mask; val_mask_mp[hashtag] = val_mask; test_mask_mp[hashtag] = test_mask; class_weight_mp[hashtag] = class_weight
        logger.info(f"hashtag={hashtag}, mask_len={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}, class_weight={class_weight[0]:.2f}:{class_weight[1]:.2f}")

    save_pickle({
        "label": labels_mp, "train_mask": train_mask_mp, "val_mask": val_mask_mp, "test_mask": test_mask_mp, "class_weight": class_weight_mp,
    }, tag2label_mask_cw_mp_filepath)

train_dict_keys, valid_dict_keys, test_dict_keys = split_cascades(preprocess_timelines, args.train_ratio, args.valid_ratio)
logger.info(f"train={len(train_dict_keys)}, valid={len(valid_dict_keys)}, test={len(test_dict_keys)}")

# $3 Stage: Model Preparation
n_user = g.vcount()
n_units = [int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]
nb_classes = 2

if args.model == 'densegat':
    # edges = g.get_edgelist() + [(i,i) for i in range(len(user_nodes))]
    diffusion_graph = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/diffusion_graph.data")
    edges = diffusion_graph[sorted(diffusion_graph.keys())[-1]].edge_index.tolist()
    edges = [(l1,l2) for l1,l2 in zip(edges[0], edges[1])] + [(i,i) for i in range(len(user_nodes))]
    adj = create_sparsemat_from_edgelist(edges, m=len(user_nodes), n=len(user_nodes))
    adj = get_sparse_tensor(adj.tocoo())

    static_user_feat = np.concatenate((structural_feat,three_sort_feat),axis=1)
    static_emb = torch.FloatTensor(static_user_feat)
    dynamic_embs = [torch.FloatTensor(deepwalk_feat)] if not args.use_tweet_feat else [torch.FloatTensor(deepwalk_feat), torch.FloatTensor(tweet_features)]
    
    static_nfeat = static_emb.shape[1]
    dynamic_nfeats = [emb.shape[1] for emb in dynamic_embs]
    n_units = [int(x) for x in args.hidden_units.strip().split(",")] + [nb_classes]
    n_heads = n_heads + [1]

    model = DenseSparseGAT(static_nfeat=static_nfeat, dynamic_nfeats=dynamic_nfeats, n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization,)
    logger.info(f"adj={adj.shape}, static_nfeats={static_nfeat}, dynamic_nfeats={dynamic_nfeats}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")
    
    if args.cuda:
        adj = adj.to(args.gpu)
        static_emb = static_emb.to(args.gpu)
        dynamic_embs = [emb.to(args.gpu) for emb in dynamic_embs]
        model = model.to(args.gpu)

elif args.model == 'heteredgegat':
    edges = g.get_edgelist() + [(i,i) for i in range(len(user_nodes))]
    adj = create_sparsemat_from_edgelist(edges, m=len(user_nodes), n=len(user_nodes))
    adj = get_sparse_tensor(adj.tocoo())

    # Building Topic-Enhanced Mats...
    classid2simmat_filepath = f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/deperacated/tweet-embedding/llm-topic/classid2simmat_minuser{args.min_user_participate}_windowsize{args.window_size}.pkl"
    tagid2classids_filepath = f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/deperacated/tweet-embedding/llm-topic/tagid2classids_minuser{args.min_user_participate}_windowsize{args.window_size}.pkl"
    if os.path.exists(classid2simmat_filepath) and os.path.exists(tagid2classids_filepath):
        classid2simmat = load_pickle(classid2simmat_filepath)
        tagid2classids = load_pickle(tagid2classids_filepath)
    else:
        TOPIC_PRETRAINED_MODELNAME = 'tweet-topic-21-multi'
        labels_aggby_timeline = load_pickle(os.path.join(DATA_ROOTPATH, f"HeterGAT/deperacated/tweet-embedding/llm-topic/topic_labels_llm_normtexts_aggby_timeline_model_{TOPIC_PRETRAINED_MODELNAME}.pkl"))
        classid2simedges, tagid2classids = build_heteredge_mats(timelines=timelines, preprocess_timelines_keys=list(preprocess_timelines.keys()), labels_aggby_timeline=labels_aggby_timeline, window_size=args.window_size, n_component=args.n_component)
        classid2simmat = {}
        for class_id, simedges in classid2simedges.items():
            extend_adj = create_sparsemat_from_edgelist(edges+simedges, m=n_user, n=n_user)
            extend_adj = get_sparse_tensor(extend_adj.tocoo())
            classid2simmat[class_id] = extend_adj
        save_pickle(classid2simmat, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/deperacated/tweet-embedding/llm-topic/classid2simmat_minuser{args.min_user_participate}_windowsize{args.window_size}.pkl")
        save_pickle(tagid2classids, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/deperacated/tweet-embedding/llm-topic/tagid2classids_minuser{args.min_user_participate}_windowsize{args.window_size}.pkl")
    logger.info(f"Completed Calculating classid2simmat map={len(classid2simmat)}...")

    if args.use_random_multiedge:
        # random_classid2simmat = {}
        # for classid, simmat in classid2simmat.items():
        #     n_simedge = simmat.coalesce().indices().shape[1]-adj.coalesce().indices().shape[1]
        #     logger.info(n_simedge)
        #     random_adj = sparse.random(n_user,n_user, density=n_simedge/n_user/n_user, format='coo')
        #     random_adj = get_sparse_tensor(random_adj)
        #     random_classid2simmat[classid] = random_adj
        #     logger.info("Completed...")
        parts = classid2simmat_filepath.split('/')
        random_classid2simmat_filepath = "/".join(parts[:-1])+f"/random_"+parts[-1]
        random_classid2simmat = load_pickle(random_classid2simmat_filepath)
        logger.info(f"Completed Calculating random_classid2simmat map={len(random_classid2simmat)}...")
        
    static_user_feat = np.concatenate((structural_feat,three_sort_feat),axis=1)
    static_emb = torch.FloatTensor(static_user_feat)
    dynamic_embs = [torch.FloatTensor(deepwalk_feat), torch.FloatTensor(tweet_features)]
    
    static_nfeat = static_emb.shape[1]
    dynamic_nfeats = [emb.shape[1] for emb in dynamic_embs]

    model = HeterEdgeSparseGAT(static_nfeat=static_nfeat, dynamic_nfeats=dynamic_nfeats, n_adj=args.n_component+1, n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization,)
    logger.info(f"adj={adj.shape}, static_nfeats={static_nfeat}, dynamic_nfeats={dynamic_nfeats}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")
    
    if args.cuda:
        adj = adj.to(args.gpu)
        static_emb = static_emb.to(args.gpu)
        dynamic_embs = [emb.to(args.gpu) for emb in dynamic_embs]
        model = model.to(args.gpu)

elif args.model == 'tan':
    opt = Option()
    opt.user_size = train_data.user_size
    new_d = {'opt':opt}
    model = TAN(opt)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def evaluate(epoch, dict_keys, best_thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()

    loss, correct, total, prec, rec, f1 = 0., 0., 0., 0., 0., 0.
    y_true, y_pred, y_score, series_y_true, series_y_prob = [], [], [], [], []
    for hashtag in dict_keys:
        if hashtag not in labels_mp or (args.model == 'heteredgegat' and hashtag not in tagid2classids): continue
        cascade_users = [elem[0] for elem in preprocess_timelines[hashtag]]

        labels, train_mask, val_mask, test_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag], class_weight_mp[hashtag]
        mask = train_mask + val_mask + test_mask
        if args.cuda:
            labels = labels.to(args.gpu); class_weight = class_weight.to(args.gpu)
        
        if args.model == 'densegat':
            output = model(adj, static_emb, dynamic_embs)
        elif args.model == 'heteredgegat':
            if not args.use_random_multiedge:
                hedge_adjs = [classid2simmat[classid].to(args.gpu) if args.cuda else classid2simmat[classid] for classid in tagid2classids[hashtag]]
                hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
            else:
                hedge_adjs = [random_classid2simmat[classid].to(args.gpu) if args.cuda else random_classid2simmat[classid] for classid in tagid2classids[hashtag]]
                hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
                # hedge_adjs = [adj] * (args.n_component+1)
            output = model(hedge_adjs, static_emb, dynamic_embs)
        
        loss_batch = F.nll_loss(output[mask], labels[mask], class_weight)
        y_true_cur = labels[mask].data.tolist()
        y_pred_cur = output[mask].max(1)[1].data.tolist()
        y_score_cur = output[mask][:, 1].data.tolist()
        y_true += y_true_cur; y_pred += y_pred_cur; y_score += y_score_cur

        alpha = mask.sum().item()
        loss += alpha * loss_batch.item()
        correct += alpha * np.mean(np.array(y_true_cur) == np.array(y_pred_cur))
        total += alpha
        
        series_y_true.append([cascade_users[-1]])
        y_prob = np.zeros(n_user)
        y_prob[mask] = y_score_cur
        series_y_prob.append(y_prob)
    
    model.train()

    if best_thr is not None:
        logger.info("using threshold %.4f", best_thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > best_thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f Acc: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f", log_desc, loss / total, correct / total, auc, prec, rec, f1)
    
    scores = compute_metrics(series_y_prob, series_y_true)
    logger.info(f"MRR={scores['MRR']}, hits@10={scores['hits@10']}, map@10={scores['map@10']}, hits@50={scores['hits@50']}, map@100={scores['map@100']}, hits@100={scores['hits@100']}, map@100={scores['map@100']},")
    
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'acc', correct / total, epoch+1);        writer.add_scalar(log_desc+'mrr', scores['MRR'], epoch+1)
    writer.add_scalar(log_desc+'hits@10', scores['hits@10'], epoch+1);  writer.add_scalar(log_desc+'map@10', scores['map@10'], epoch+1)
    writer.add_scalar(log_desc+'hits@50', scores['hits@50'], epoch+1);  writer.add_scalar(log_desc+'map@50', scores['map@50'], epoch+1)
    writer.add_scalar(log_desc+'hits@100', scores['hits@100'], epoch+1);writer.add_scalar(log_desc+'map@100', scores['map@100'], epoch+1)
    writer.add_scalar(log_desc+'auc', auc, epoch+1);                    writer.add_scalar(log_desc+'f1', f1, epoch+1)
    writer.add_scalar(log_desc+'prec', prec, epoch+1);                  writer.add_scalar(log_desc+'rec', rec, epoch+1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs + 1e-10)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None

def train(epoch, dict_keys, log_desc='train_'):
    model.train()
    loss, total = 0., 0.
    for hashtag in sorted(dict_keys):
        if hashtag not in labels_mp or (args.model == 'heteredgegat' and hashtag not in tagid2classids): continue

        labels, train_mask, val_mask, test_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag], class_weight_mp[hashtag]
        mask = train_mask + val_mask + test_mask
        if args.cuda:
            labels = labels.to(args.gpu); class_weight = class_weight.to(args.gpu)
        
        optimizer.zero_grad()
        if args.model == 'densegat':
            output = model(adj, static_emb, dynamic_embs)
        elif args.model == 'heteredgegat':
            if not args.use_random_multiedge:
                hedge_adjs = [classid2simmat[classid].to(args.gpu) if args.cuda else classid2simmat[classid] for classid in tagid2classids[hashtag]]
                hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
            else:
                hedge_adjs = [random_classid2simmat[classid].to(args.gpu) if args.cuda else random_classid2simmat[classid] for classid in tagid2classids[hashtag]]
                hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
                # hedge_adjs = [adj] * (args.n_component+1)
            output = model(hedge_adjs, static_emb, dynamic_embs)
        loss_train = F.nll_loss(output[mask], labels[mask], class_weight)
        alpha = mask.sum().item()
        loss  += alpha * loss_train.item()
        total += alpha
        loss_train.backward()
        optimizer.step()
    logger.info(f"train loss in this epoch {loss / total}, GPU Memory Usage = {check_gpu_memory_usage(int(args.gpu[-1]))} MiB")
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)

    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_dict_keys, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_dict_keys, best_thr=best_thr, log_desc='test_')
        save_model(epoch, args, model, optimizer)

t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch, train_dict_keys)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(args.epochs, valid_dict_keys, return_best_thr=True, log_desc='valid_')
logger.info("testing...")
evaluate(args.epochs, test_dict_keys, best_thr=best_thr, log_desc='test_')
