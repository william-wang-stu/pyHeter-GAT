import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
from src.utils import extend_wholegraph, load_pickle, save_pickle, unique_cascades, create_adjmat_from_edgelist, load_labels, DATA_ROOTPATH, Ntimestage
from lib.utils import get_sparse_tensor
from lib.log import logger
import argparse
import shutil
# from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import tensorboard_logger
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import *

def load_dataset(args, g):
    ut_mp = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/text/utmp_groupbystage.p"))
    hadjs, feats = extend_wholegraph(g=g, ut_mp=ut_mp, stage=args.stage, tweet_per_user=args.tweet_per_user, sparse_graph=False)
    feature_dim = feats.shape[1]
    num_user = g.vcount()
    # hadjs = [get_sparse_tensor(hadj.tocoo()) for hadj in hadjs]
    # feats = torch.FloatTensor(feats)
    return hadjs, feats, feature_dim, num_user

parser = argparse.ArgumentParser()
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--min-influence', type=int, default=100, help='Min Influence Length')
parser.add_argument('--attn-dropout', type=float, default=0.1, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8,1", help="Heads in each layer, splitted with comma")
parser.add_argument('--tweet-per-user', type=int, default=1, help="Tweets Per User (20, 40, 100)")
parser.add_argument('--sample-ratio', type=int, default=1, help="Sampling Ratio (1~inf)")
parser.add_argument('--stage', type=int, default=Ntimestage-1, help="Time Stage (0~Ntimestage-1)")
parser.add_argument('--train-ratio', type=float, default=60, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=20, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
parser.add_argument('--gpu', type=str, default="cuda:0", help="Select GPU")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

g  = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subg_dp_20_100_ratio_35_20_2.p")) # Total 44896 User Nodes
df = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/subdf_dp_20_100_ratio_35_20_2.p"))
df = unique_cascades(df)
df = {hashtag:cascades for hashtag,cascades in df.items() if len(cascades)>=args.min_influence}
hadjs, feats, feature_dim, n_user = load_dataset(args, g)
fea_list = [torch.transpose(torch.from_numpy(feats[np.newaxis]),2,1) for _ in range(3)]
# n_units = [feature_dim]+[int(x) for x in args.hidden_units.strip().split(",")]
n_units = [int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]

nb_nodes = hadjs[0].shape[0]
ft_size  = feats.shape[1]
adj_list = [hadj[np.newaxis] for hadj in hadjs]
biases_list = [torch.transpose(torch.from_numpy(adj_to_bias(adj, [nb_nodes], nhood=1)),2,1) for adj in adj_list]

# save_pickle(fea_list, "data/fea_list.p")
# save_pickle(biases_list, "data/biases_list.p")
# fea_list = load_pickle("data/fea_list.p")
# biases_list = load_pickle("data/biases_list.p")
fea_list = [fea.to(args.gpu) for fea in fea_list]
biases_list = [biases.to(args.gpu) for biases in biases_list]
logger.info(f"biases_list={len(biases_list)}.{biases_list[0].shape}, fea_list={len(fea_list)}.{fea_list[0].shape}, n_user={n_user}, n_units={n_units}, n_heads={n_heads}")

# adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, my_data = load_data_dblp(path=os.path.join(DATA_ROOTPATH, "HeterNet/ACM/ACM3025.mat"))

tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_mininf%d_t%d' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.min_influence, args.tweet_per_user)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.configure(tensorboard_log_dir)
# writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

# dataset = 'acm'
# featype = 'fea'
# checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
# print('model: {}'.format(checkpt_file))

model = HeteGAT_multi(inputs_list=fea_list,nb_classes=2,nb_nodes=nb_nodes,attn_drop=0.5,
    ffd_drop=0.0,bias_mat_list=biases_list,hid_units=n_units,n_heads=n_heads,
    activation=nn.ELU(),residual=False)
model.to(args.gpu)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(params=model.parameters(),lr=args.lr,betas=(0.9,0.99),weight_decay=0.0)

# NOTE: memo labels and masks pre
labels_mp, train_mask_mp, val_mask_mp, test_mask_mp = {}, {}, {}, {}
for hashtag, cascades in df.items():
    labels, train_mask, val_mask, test_mask, nb_classes, _ = load_labels(args=args, g=g, cascades=cascades)
    labels_mp[hashtag] = labels; train_mask_mp[hashtag] = train_mask; val_mask_mp[hashtag] = val_mask; test_mask_mp[hashtag] = test_mask
    logger.info(f"hashtag={hashtag:}, mask_len={train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}, nb_classes={nb_classes}")

# train_my_labels = torch.from_numpy(train_my_labels).long().to(args.gpu)
# val_my_labels = torch.from_numpy(val_my_labels).long().to(args.gpu)
# test_my_labels = torch.from_numpy(test_my_labels).long().to(args.gpu)

# train_mask = np.where(train_mask == 1)[0]
# val_mask = np.where(val_mask == 1)[0]
# test_mask = np.where(test_mask == 1)[0]
# train_mask = torch.from_numpy(train_mask).to(args.gpu)
# val_mask = torch.from_numpy(val_mask).to(args.gpu)
# test_mask = torch.from_numpy(test_mask).to(args.gpu)

def main():
    for epoch in range(1,args.epochs):
        train(epoch)
        test(epoch, 'val_', 'val')
    test(epoch, 'test_', 'test')

def train(epoch, log_desc='train_'):
    model.train()
    correct = 0.
    loss, total = 0., 0.
    for hashtag in sorted(df.keys()):
        labels, train_mask = labels_mp[hashtag], train_mask_mp[hashtag]
        if args.cuda:
            labels = labels.to(args.gpu)
        
        optimizer.zero_grad()
        outputs = model(fea_list)
        # train_mask_outputs = torch.index_select(outputs,0,train_mask)
        _, preds = torch.max(outputs[train_mask].data,1)
        loss_train = criterion(outputs[train_mask],labels[train_mask])

        alpha = train_mask.sum().item()
        loss  += alpha * loss_train.item()
        total += alpha
        correct += torch.sum(preds == labels[train_mask])
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f, acc=%f", loss / total, correct / total)
    tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    tensorboard_logger.log_value(log_desc+'acc', correct / total, epoch+1)

def test(epoch, log_desc, mode):
    model.eval()
    correct, loss, total = 0., 0., 0.
    y_pred, y_true, y_score = [], [], []
    with torch.no_grad():
        for hashtag in sorted(df.keys()):
            labels, val_mask, test_mask = labels_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag]
            if args.cuda:
                labels = labels.to(args.gpu)
            
            outputs = model(fea_list)
            if mode == 'val':
                mask = val_mask
                alpha = val_mask.sum().item()
            elif mode == 'test':
                mask = test_mask
                alpha = test_mask.sum().item()
            y_pred_cur  = outputs[mask].max(1)[1]
            y_true_cur  = labels[mask]
            y_pred += y_pred_cur.data.tolist(); y_true += y_true_cur.data.tolist()
            y_score += outputs[mask][:,1].data.tolist()
            loss_batch = criterion(outputs[mask], labels[mask])

            correct += torch.sum(y_pred_cur == y_true_cur)
            loss  += alpha * loss_batch.item()
            total += alpha

    model.train()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f Acc: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f", log_desc, loss / total, correct / total, auc, prec, rec, f1)
    tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    tensorboard_logger.log_value(log_desc+'acc', correct / total, epoch+1)
    tensorboard_logger.log_value(log_desc+'auc', auc, epoch+1)
    tensorboard_logger.log_value(log_desc+'prec', prec, epoch+1)
    tensorboard_logger.log_value(log_desc+'rec', rec, epoch+1)
    tensorboard_logger.log_value(log_desc+'f1', f1, epoch+1)

main()
