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
from utils import load_pickle, save_pickle, DATA_ROOTPATH
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
from tensorboard_logger import tensorboard_logger

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}")

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def gen_labels(stage, graph_vcount=44896):
    df = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_df.p"))
    affected_users = [set() for _ in range(8)]

    for _, cascades in df.items():
        if len(cascades) == 0:
            continue
        time_span = int((cascades[-1][1]-cascades[0][1])/8)
        for tidx in range(8):
            affected_users[tidx] |= set([elem[0] for elem in cascades if elem[1]>=cascades[0][1]+time_span*tidx and elem[1]<cascades[0][1]+time_span*(tidx+1)])

    # for tidx in range(8):
    #     logger.info(f"{tidx}={len(affected_users[tidx])}")

    labels = get_binary_mask(graph_vcount, list(affected_users[stage]))
    return labels

# NOTE: 
# vertex_ids[:,-1], labels -> ids=np.unique(vertex_ids[:,-1]) -> train_/valid_/test_ids
def load_dataset(args):
    # 1. Loading hadjs and feats
    # hadjs = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_hadjs.p"))
    hadjs = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_hadjs_selfloop_max20tweet.p"))
    feats = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_feats_max20tweet.p"))
    feature_dim = feats.shape[1]
    num_user = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deg_le483_subgraph.p")).vcount()

    hadjs = [get_sparse_tensor(hadj.tocoo()) for hadj in hadjs]
    feats = torch.FloatTensor(feats)

    # 2. Loading labels and vertex_ids
    labels = gen_labels(stage=args.label, graph_vcount=num_user)

    float_mask = np.zeros(len(labels))
    for label in np.unique(labels):
        ids = np.where(labels == label)[0]
        if args.shuffle:
            float_mask[ids] = np.random.permutation(np.linspace(0,1,len(ids)))
        else:
            float_mask[ids] = np.linspace(0,1,len(ids))
    
    train_ids = np.where(float_mask<=args.train_ratio/100)[0]
    val_ids   = np.where((float_mask>args.train_ratio/100) & (float_mask<=(args.train_ratio+args.valid_ratio)/100))[0]
    test_ids  = np.where(float_mask>(args.train_ratio+args.valid_ratio)/100)[0]
    logger.info(f"train/valid/test={len(train_ids)},{len(val_ids)},{len(test_ids)}")

    train_mask = get_binary_mask(num_user, train_ids)
    val_mask   = get_binary_mask(num_user, val_ids)
    test_mask  = get_binary_mask(num_user, test_ids)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
    
    nb_classes = np.unique(labels).shape[0]
    class_weight = torch.FloatTensor(len(labels) / (nb_classes*np.bincount(labels))) if args.class_weight_balanced else torch.ones(nb_classes)
    
    return hadjs, feats, feature_dim, labels, train_mask, val_mask, test_mask, num_user, nb_classes, class_weight

parser = argparse.ArgumentParser()
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='hetersparsegat', help="available options are ['hetersparsegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=False, help="Shuffle dataset")
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.5, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--label', type=int, default=4, help="Select Active Users From Which Stage (0~7)")
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

tensorboard_log_dir = 'tensorboard/%s_%s_lr%f_hid%s:%s' % (args.model, args.tensorboard_log, args.lr, args.hidden_units, args.heads)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.configure(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

hadjs, feats, feature_dim, labels, train_mask, val_mask, test_mask, n_user, nb_classes, class_weight = load_dataset(args)
n_units = [feature_dim]+[int(x) for x in args.hidden_units.strip().split(",")]+[nb_classes]
n_heads = [int(x) for x in args.heads.strip().split(",")]+[1]
logger.info(f"hadjs=[{len(hadjs)},{hadjs[0].shape}], feats={feats.shape}, labels={labels.shape}, n_user={n_user}")
logger.info(f"class_weight={class_weight[0]:.2f}:{class_weight[1]:.2f}, nb_classes={nb_classes}, n_units={n_units}, n_heads={n_heads}")

if args.model == 'hetersparsegat':
    model = HetersparseGAT(n_user=n_user, nb_node_kinds=2, nb_classes=nb_classes, n_units=n_units, n_heads=n_heads, 
        attn_dropout=args.attn_dropout, dropout=args.dropout, use_pretrained_emb=False)
logger.info(f"model: {model}")

if args.cuda:
    for idx in range(len(hadjs)):
        hadjs[idx] = hadjs[idx].to(args.gpu)
    feats = feats.to(args.gpu)
    labels = labels.to(args.gpu)
    model.to(args.gpu)
    class_weight = class_weight.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def evaluate(epoch, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()

    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    
    if args.model == 'hetersparsegat':
        output = model(hadjs, feats)
    
    if thr is None: # valid
        loss = F.nll_loss(output[val_mask], labels[val_mask], class_weight)
        y_true  = labels[val_mask].data.tolist()
        y_pred  = output[val_mask].max(1)[1].data.tolist()
        y_score = output[val_mask][:, 1].data.tolist()
    else: # test
        loss = F.nll_loss(output[test_mask], labels[test_mask],  class_weight)
        y_true  = labels[test_mask].data.tolist()
        y_pred  = output[test_mask].max(1)[1].data.tolist()
        y_score = output[test_mask][:, 1].data.tolist()

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f", log_desc, loss, auc, prec, rec, f1)
    tensorboard_logger.log_value(log_desc + 'loss', loss, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'auc', auc, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'prec', prec, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'rec', rec, epoch + 1)
    tensorboard_logger.log_value(log_desc + 'f1', f1, epoch + 1)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))
        return best_thr
    else:
        return None

def train(epoch, log_desc='train_'):
    model.train()
    optimizer.zero_grad()
    if args.model == 'hetersparsegat':
        output = model(hadjs, feats)
    # logger.info(f"output={output[:-1]}")
    loss_train = F.nll_loss(output[train_mask], labels[train_mask], class_weight)
    loss_train.backward()
    optimizer.step()
    logger.info("train loss in this epoch %f", loss_train)
    tensorboard_logger.log_value('train_loss', loss_train, epoch + 1)

    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, thr=best_thr, log_desc='test_')

# Train model
t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(args.epochs, return_best_thr=True, log_desc='valid_')

# Testing
logger.info("testing...")
evaluate(args.epochs, thr=best_thr, log_desc='test_')
