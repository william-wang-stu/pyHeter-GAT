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
from src.model import HeterSparseGAT, HyperGAT, HyperGATWithHeterSparseGAT
from src.model_batchdensegat import BatchDenseGAT, BatchGAT
from src.data_loader import InfluenceDataSet, ChunkSampler
import numpy as np
import argparse
import shutil
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='hypergat', help="available options are ['batchdensegat','hetersparsegat','hypergat','hypergatwithhetersparsegat']")
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
parser.add_argument('--fine-tune', action='store_true', default=False, help="Fine Tune Model Global Embeddings")
parser.add_argument('--sparse-data', action='store_true', default=True, help="Use Sparse Data and Model (Only Valid When model='hypergat')")
parser.add_argument('--wo-user-centralized-net', action='store_true', default=False, help="Ablation Study w/o user-centralized-network")
parser.add_argument('--wo-tweet-centralized-net', action='store_true', default=False, help="Ablation Study w/o tweet-centralized-network")
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=1024, help="Batch size")
# [default] hetersparsegat: 3e-3, hypergat: 3e-3
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--unified-dim', type=int, default=64, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--stage', type=int, default=Ntimestage-1, help="Time Stage (0~Ntimestage-1)")
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--train-ratio', type=float, default=75, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=12.5, help="Validation ratio (0, 100)")
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

# structural_feats = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/vertex_feature_subgle483.npy"))
# struc_cascade_feats = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/user_features_avg.p"))
# deepwalk_feats = load_w2v_feature(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deepwalk/deepwalk_added.emb_64"), max_idx=user_nodes[-1]+1)
# fn = lambda val,ind: val if len(val)==len(ind) else val[ind]
# user_features = np.concatenate((
#     fn(normalize(structural_feats),ind=user_nodes), 
#     fn(normalize(struc_cascade_feats),ind=user_nodes), 
#     fn(normalize(deepwalk_feats),ind=user_nodes)), axis=1)

vertex_feat = load_pickle("/remote-home/share/dmb_nas/wangzejian/digg/digg/vertex_feature.npy")
vertex_feat = preprocessing.scale(vertex_feat)
deepwalk_feat = load_w2v_feature("/remote-home/share/dmb_nas/wangzejian/digg/digg/deepwalk.emb_64", max_idx=vertex_feat.shape[0]-1)

# TODO: consider args.stage when choosing tweet neighbors
# if isinstance(args.stage, int) and 0<=args.stage<=7:
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))
#     user_tweet_mp = user_tweet_mp[args.stage]
# else:
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))

# if args.embedding_method == 'lda':
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))
#     tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model/doc2topic_k25_maxiter50.p"))
# elif args.embedding_method == 'bertopic':
#     user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/ut_mp_filter_lt2words_processedforbert_subg.pkl"))
#     tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/tweet-embedding/bertopic/topic_distribution_remove_hyphen_reduce_auto.pkl"))
# if not isinstance(tweet_features, np.ndarray):
#     tweet_features = np.array(tweet_features)

# logger.info(f"user-feats dim={user_features.shape[1]}, tweet-feats dim={tweet_features.shape[1]}")

# tensorboard-logger Preparation
tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_batch%d_t%d_drop%f_attndrop%f' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.batch, args.tweet_per_user, args.dropout, args.attn_dropout)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# Stage: Data Processing
if args.model == 'batchdensegat':
    influence_dataset = InfluenceDataSet(
        # file_dir=os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subgraph_sampling/subg_inf_5_400_deg_3_62_ego_49_neg_1_restart_0.2_1000"),
        file_dir="/remote-home/share/dmb_nas/wangzejian/digg/digg",
        seed=args.seed, shuffle=args.shuffle, model='gat')
# elif args.model == 'batchgat':
#     influence_dataset = InfluenceDataSet2(
#         file_dir=os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subgraph_sampling/subg_inf_5_400_deg_3_62_ego_49_neg_1_restart_0.2_1000"),
#         # file_dir="/remote-home/share/dmb_nas/wangzejian/digg/digg",
#         embedding_dim=64, seed=args.seed, shuffle=args.shuffle, model='gat')
nb_classes = influence_dataset.get_num_class()
class_weight = influence_dataset.get_class_weight()
N = len(influence_dataset)

train_start,  valid_start, test_start = 0, int(N*args.train_ratio/100), int(N*(args.train_ratio+args.valid_ratio)/100)
train_loader = DataLoader(influence_dataset, batch_size=args.batch, sampler=ChunkSampler(valid_start-train_start, 0))
valid_loader = DataLoader(influence_dataset, batch_size=args.batch, sampler=ChunkSampler(test_start-valid_start, valid_start))
test_loader = DataLoader(influence_dataset, batch_size=args.batch, sampler=ChunkSampler(N-test_start, test_start))
logger.info(f"train_start={train_start}, valid_start={valid_start}, test_start={test_start}, train_loader={len(train_loader)}, valid_loader={len(valid_loader)}, test_loader={len(test_loader)}")

# Stage: Model Preparation
n_user = g.vcount()
n_units = [int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]

if args.model == 'batchdensegat':
    global_embs = [torch.FloatTensor(vertex_feat), torch.FloatTensor(deepwalk_feat)]
    n_feat = vertex_feat.shape[1]+deepwalk_feat.shape[1]+influence_dataset.get_influence_feature_dimension()
    n_units = [int(x) for x in args.hidden_units.strip().split(",")] + [nb_classes]
    n_heads = n_heads + [1]
    model = BatchDenseGAT(global_embs=global_embs, n_feat=n_feat, n_units=n_units, n_heads=n_heads, shape_ret=(n_user,nb_classes),
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, norm_mask=[0,1], fine_tune=args.fine_tune)
    logger.info(f"n_user={n_user}, n_feat={n_feat}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")
    
# elif args.model == 'batchgat':
#     feature_dim = influence_dataset.get_feature_dimension()
#     # n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")]
#     n_units = [feature_dim] + [int(x) for x in args.hidden_units.strip().split(",")] + [nb_classes]
#     n_heads = n_heads + [1]
#     model = BatchGAT(
#             # pretrained_emb=influence_dataset.get_embedding(),
#             # vertex_feature=influence_dataset.get_vertex_features(),
#             pretrained_emb=torch.FloatTensor(deepwalk_feat),
#             vertex_feature=torch.FloatTensor(vertex_feat),
#             use_vertex_feature=True,
#             n_units=n_units, n_heads=n_heads,
#             dropout=args.dropout, instance_normalization=args.instance_normalization, fine_tune=args.fine_tune)
#     logger.info(f"n_user={n_user}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")

if args.cuda:
    model.to(args.gpu)
    class_weight = class_weight.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# stopper = EarlyStopping(patience=args.patience)

def save_model(epoch, args, model, optimizer):
    state = {
        "epoch": epoch,
        "args": args,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_filepath = os.path.join(DATA_ROOTPATH, f"HeterGAT/basic/training/ckpt_epoch{epoch}_model_{args.model}.pkl")
    torch.save(state, save_filepath)
    # help release GPU memory
    del state
    logger.info(f"Save State To Path={save_filepath}... Checkpoint Epoch={epoch}")

def load_model(filename, model):
    ckpt_state = torch.load(filename, map_location='cpu')
    model.load_state_dict(ckpt_state['model'])
    # optimizer.load_state_dict(ckpt_state['optimizer'])
    logger.info(f"Load State From Path={filename}... Checkpoint Epoch={ckpt_state['epoch']}")

def evaluate(epoch, data_loader, best_thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()

    loss, correct, total, prec, rec, f1 = 0., 0., 0., 0., 0., 0.
    y_true, y_pred, y_score, thrs = [], [], [], {}
    for i, (graphs, influence_feats, labels, vertices) in enumerate(data_loader):
        bs = graphs.size(0)
        if args.cuda:
            graphs, influence_feats, labels, vertices = graphs.to(args.gpu), influence_feats.to(args.gpu), labels.to(args.gpu), vertices.to(args.gpu)

        if args.model == "batchdensegat":
            output = model(graphs, vertices, influence_feats)
            output = output[:, -1, :]
        # elif args.model == 'batchgat':
        #     output = model(influence_feats, vertices, graphs)
        #     output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        y_true_cur = labels.data.tolist()
        y_pred_cur = output.max(1)[1].data.tolist()

        loss += bs * loss_batch.item()
        correct += bs * np.mean(np.array(y_true_cur) == np.array(y_pred_cur))
        total += bs

        y_true += y_true_cur
        y_pred += y_pred_cur
        y_score += output[:, 1].data.tolist()

    model.train()

    if best_thr is not None:
        logger.info("using threshold %.4f", best_thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > best_thr] = 1

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

def train(epoch, data_loader, log_desc='train_'):
    model.train()
    loss, total = 0., 0.
    for i, (graphs, influence_feats, labels, vertices) in enumerate(data_loader):
        bs = graphs.size(0)
        if args.cuda:
            graphs, influence_feats, labels, vertices = graphs.to(args.gpu), influence_feats.to(args.gpu), labels.to(args.gpu), vertices.to(args.gpu)
        
        optimizer.zero_grad()
        if args.model == 'batchdensegat':
            output = model(graphs, vertices, influence_feats)
            output = output[:,-1,:]
        # elif args.model == 'batchgat':
        #     output = model(influence_feats, vertices, graphs)
        #     output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss  += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)

    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, best_thr=best_thr, log_desc='test_')
        save_model(epoch, args, model, optimizer)

t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch, train_loader)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(args.epochs, valid_loader, return_best_thr=True, log_desc='valid_')
logger.info("testing...")
evaluate(args.epochs, test_loader, best_thr=best_thr, log_desc='test_')
