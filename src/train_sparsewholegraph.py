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
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import get_sparse_tensor
from lib.log import logger
from utils import *
from model import HetersparseGAT, HyperGraphAttentionNetwork
import numpy as np
import argparse
import shutil
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
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
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='hypergat', help="available options are ['hetersparsegat','hypergat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
# [default] hetersparsegat: 3e-3, hypergat: 3e-3
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.1, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--stage', type=int, default=Ntimestage-1, help="Time Stage (0~Ntimestage-1)")
parser.add_argument('--min-influence', type=int, default=100, help='Min Influence Length')
parser.add_argument('--tweet-per-user', type=int, default=10, help="Tweets Per User (20, 40, 100)")
parser.add_argument('--sample-ratio', type=int, default=1, help="Sampling Ratio (1~inf)")
parser.add_argument('--selected-tags', type=str, default="all", help="Agg On Some Part of Hastags")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--train-ratio', type=float, default=60, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=20, help="Validation ratio (0, 100)")
parser.add_argument('--instance-normalization', action='store_true', default=False, help="Enable instance normalization")
parser.add_argument('--sparse-data', action='store_true', default=True, help="Use Sparse Data and Model (Only Valid When model='hypergat')")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
parser.add_argument('--sota-test', action='store_true', default=True, help="Use Prepared Sota-Test-Dataset if set true")
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
    df = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_df.p"))
else:
    # Sota-Test uses a down-sampled version of normal dataset, and corresponding down-sample-ratio is clarified in its filename
    g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subg_dp_20_100_ratio_35_20_2.p"))
    df = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subdf_dp_20_100_ratio_35_20_2.p"))
user_nodes = g.vs["label"]

if isinstance(args.stage, int) and 0<=args.stage<=7:
    user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))
    user_tweet_mp = user_tweet_mp[args.stage]
else:
    user_tweet_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/usertweet_mp.p"))

structural_temporal_feats = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/user_features_avg.p"))
deepwalk_feats = load_w2v_feature(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deepwalk/deepwalk_added.emb_64"), 208894)
user_features = np.concatenate((structural_temporal_feats[user_nodes], deepwalk_feats[user_nodes]), axis=1)
tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/lda-model/doc2topic_k25_maxiter50.p"))
if not isinstance(tweet_features, np.ndarray):
    tweet_features = np.array(tweet_features)
logger.info(f"user-feats dim={user_features.shape[1]}, tweet-feats dim={tweet_features.shape[1]}")

# tensorboard-logger Preparation
tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_mininf%d_t%d' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.min_influence, args.tweet_per_user)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# Stage: Data Processing
df = unique_cascades(df) # remove duplicate appearances for single user in some particular hashtag
df = {hashtag:cascades for hashtag,cascades in df.items() if len(cascades)>=args.min_influence} # remove some hashtags with too few user occurences

labels_mp, train_mask_mp, val_mask_mp, test_mask_mp, class_weight_mp = {}, {}, {}, {}, {}
selected_tags = list(df.keys()) if args.selected_tags == "all" else [int(x) for x in args.selected_tags.split(',')]
for hashtag, cascades in df.items():
    if hashtag not in selected_tags:
        continue
    labels, train_mask, val_mask, test_mask, nb_classes, class_weight = load_labels(args=args, g=g, cascades=cascades)
    labels_mp[hashtag] = labels; train_mask_mp[hashtag] = train_mask; val_mask_mp[hashtag] = val_mask; test_mask_mp[hashtag] = test_mask; class_weight_mp[hashtag] = class_weight
    logger.info(f"hashtag={hashtag}, mask_len={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}, nb_classes={nb_classes}, class_weight={class_weight[0]:.2f}:{class_weight[1]:.2f}")

# Stage: Model Preparation
n_user = g.vcount()
n_units = [int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]

if args.model == 'hetersparsegat':
    hadjs, feats = extend_wholegraph(g=g, ut_mp=user_tweet_mp, initial_feats=[user_features, tweet_features], tweet_per_user=args.tweet_per_user)
    hadjs = [get_sparse_tensor(hadj.tocoo()) for hadj in hadjs]
    feats = torch.FloatTensor(feats)
    model = HetersparseGAT(n_user=n_user, nb_node_kinds=2, nb_classes=nb_classes, n_units=[feats.shape[1]]+n_units, n_heads=n_heads, 
        attn_dropout=args.attn_dropout, dropout=args.dropout, use_pretrained_emb=False)
    logger.info(f"hadjs={len(hadjs)}*{hadjs[0].shape}, feats={feats.shape}, n_user={n_user}, n_units={n_units}, n_heads={n_heads}")

    if args.cuda:
        hadjs = [hadj.to(args.gpu) for hadj in hadjs]
        feats = feats.to(args.gpu)
        model.to(args.gpu)

elif args.model == 'hypergat':
    tw_nodes, tw_edges, tw_feats = tweet_centralized_process(homo_g=g, user_tweet_mp=user_tweet_mp, tweet_features=tweet_features, clustering_algo='mbk')
    user_edges = sum([g.get_edgelist(), [(elem,elem) for elem in range(len(g.vs))]], [])
    hadjs = [
        create_sparsemat_from_edgelist(edgelist=user_edges, m=n_user, n=n_user) if args.sparse_data else 
            create_adjmat_from_edgelist(edgelist=user_edges, size=n_user),
        create_sparsemat_from_edgelist(edgelist=tw_edges, m=len(tw_nodes), n=len(tw_nodes)) if args.sparse_data else
            create_adjmat_from_edgelist(edgelist=tw_edges, size=len(tw_nodes)),
    ]
    hadjs = [get_sparse_tensor(hadj.tocoo()) if args.sparse_data else torch.BoolTensor(hadj) for hadj in hadjs]
    hembs = [torch.FloatTensor(user_features), torch.FloatTensor(tw_feats),]
    model = HyperGraphAttentionNetwork(n_user=n_user, heter_vecspace_dims=[user_features.shape[1], tw_feats.shape[1]], nb_classes=nb_classes, 
        n_units=n_units, n_heads=n_heads, attn_dropout=args.attn_dropout, dropout=args.dropout,
        instance_normalization=args.instance_normalization, sparse_data=args.sparse_data,)
    logger.info(f"hadjs={len(hadjs)}*{hadjs[0].shape}, feats={hembs[0].shape}:{hembs[1].shape}, n_user={n_user}, n_units={n_units}, n_heads={n_heads}")

    if args.cuda:
        hadjs = [hadj.to(args.gpu) for hadj in hadjs]
        hembs = [hemb.to(args.gpu) for hemb in hembs]
        model.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# stopper = EarlyStopping(patience=args.patience)

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
    for hashtag in sorted(selected_tags):
        labels, val_mask, test_mask, class_weight = labels_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag], class_weight_mp[hashtag]
        if args.cuda:
            labels, class_weight = labels.to(args.gpu), class_weight.to(args.gpu)
        
        if args.model == 'hetersparsegat':
            output = model(hadjs, feats)
        elif args.model == 'hypergat':
            output = model(hadjs, hembs)
        
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
        correct += alpha * np.sum(np.array(y_true_cur) == np.array(y_pred_cur))
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
    for hashtag in sorted(selected_tags):
        labels, train_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], class_weight_mp[hashtag]
        if args.cuda:
            labels, class_weight = labels.to(args.gpu), class_weight.to(args.gpu)
        
        optimizer.zero_grad()
        if args.model == 'hetersparsegat':
            output = model(hadjs, feats)
        elif args.model == 'hypergat':
            output = model(hadjs, hembs)
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
