"""
总体思路: 
1. 输入数据: 异质图网络, 异质图节点特征
-   使用子网训练测试, 每个子网Sample上仅包括User节点, 这里为子图中的每个节点随机选取若干(N=20)个Tweet节点一并用作测试
    P.S. 注意此时子网大小为50Users+50*30Tweets=1550
2. 网路
3. 训练测试
"""
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import get_sparse_tensor
from lib.log import logger
from utils import extend_wholegraph, EarlyStopping, load_pickle, save_pickle, DATA_ROOTPATH, Ntimestage
from model import HetersparseGAT
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

def load_dataset(args, g):
    ut_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))
    hadjs, feats = extend_wholegraph(g=g, ut_mp=ut_mp, stage=args.stage, tweet_per_user=args.tweet_per_user)
    feature_dim = feats.shape[1]
    num_user = g.vcount()
    hadjs = [get_sparse_tensor(hadj.tocoo()) for hadj in hadjs]
    feats = torch.FloatTensor(feats)
    return hadjs, feats, feature_dim, num_user

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

def gen_pos_neg_users(g, cascades, sample_ratio, stage):
    pos_users, neg_users = set(), set()
    all_activers = set([elem[0] for elem in cascades])
    max_ts = cascades[0][1] + int((cascades[-1][1]-cascades[0][1]+Ntimestage-1)/Ntimestage) * (stage+1)
    for user, ts in cascades:
        if ts > max_ts:
            break
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

def load_labels(args, g, cascades):
    pos_users, neg_users = gen_pos_neg_users(g=g, cascades=cascades, sample_ratio=args.sample_ratio, stage=args.stage)
    num_user = g.vcount()

    # NOTE: label=1表示正样本, label=-1表示负样本, label=0表示未被选择
    labels = torch.zeros(num_user)
    labels[list(pos_users)] =  1
    labels[list(neg_users)] = -1

    float_mask = np.zeros(len(labels))
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
    
    pos_neg_labels = list(filter(lambda x: x==1 or x==-1, labels))
    nb_classes = np.unique(pos_neg_labels).shape[0]
    class_weight = torch.FloatTensor(len(pos_neg_labels) / (nb_classes*np.unique(pos_neg_labels, return_counts=True)[1])) if args.class_weight_balanced else torch.ones(nb_classes)
    labels[labels==-1] = 0
    
    return labels.long(), train_mask, val_mask, test_mask, nb_classes, class_weight

parser = argparse.ArgumentParser()
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='hetersparsegat', help="available options are ['hetersparsegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--min-influence', type=int, default=100, help='Min Influence Length')
parser.add_argument('--attn-dropout', type=float, default=0.1, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--tweet-per-user', type=int, default=10, help="Tweets Per User (20, 40, 100)")
parser.add_argument('--sample-ratio', type=int, default=1, help="Sampling Ratio (1~inf)")
parser.add_argument('--stage', type=int, default=Ntimestage-1, help="Time Stage (0~Ntimestage-1)")
parser.add_argument('--selected-tags', type=str, default="all", help="Agg On Some Part of Hastags")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--train-ratio', type=float, default=60, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=20, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
parser.add_argument('--gpu', type=str, default="cuda:1", help="Select GPU")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_subgraph.p")) # Total 44896 User Nodes
# df = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_df.p"))
g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subg_dp_20_100_ratio_35_20_2.p")) # Total 44896 User Nodes
df = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subdf_dp_20_100_ratio_35_20_2.p"))
df = unique_cascades(df)
df = {hashtag:cascades for hashtag,cascades in df.items() if len(cascades)>=args.min_influence}
hadjs, feats, feature_dim, n_user = load_dataset(args, g)
n_units = [feature_dim]+[int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]
logger.info(f"hadjs=[{len(hadjs)},{hadjs[0].shape}], feats={feats.shape}, n_user={n_user}, n_units={n_units}, n_heads={n_heads}")

if args.cuda:
    hadjs = [hadj.to(args.gpu) for hadj in hadjs]
    feats = feats.to(args.gpu)

tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_mininf%d_t%d' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.min_influence, args.tweet_per_user)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

if args.selected_tags == "all":
    selected_tags = list(df.keys())
else:
    selected_tags = [int(x) for x in args.selected_tags.split(',')]
logger.info(f"selected-tags len={len(selected_tags)}")
selected_df = {hashtag: cascades for hashtag,cascades in df.items() if hashtag in selected_tags}

# NOTE: memo labels and masks pre
labels_mp, train_mask_mp, val_mask_mp, test_mask_mp, class_weight_mp = {}, {}, {}, {}, {}
for hashtag, cascades in selected_df.items():
    labels, train_mask, val_mask, test_mask, nb_classes, class_weight = load_labels(args=args, g=g, cascades=cascades)
    labels_mp[hashtag] = labels; train_mask_mp[hashtag] = train_mask; val_mask_mp[hashtag] = val_mask; test_mask_mp[hashtag] = test_mask; class_weight_mp[hashtag] = class_weight
    logger.info(f"hashtag={hashtag:}, mask_len={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}, nb_classes={nb_classes}, class_weight={class_weight[0]:.2f}:{class_weight[1]:.2f}")

if args.model == 'hetersparsegat':
    model = HetersparseGAT(n_user=n_user, nb_node_kinds=2, nb_classes=nb_classes, n_units=n_units, n_heads=n_heads, 
        attn_dropout=args.attn_dropout, dropout=args.dropout, use_pretrained_emb=False)
# logger.info(f"model: {model}")
if args.cuda:
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
