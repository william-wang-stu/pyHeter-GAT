# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

from lib.log import logger
from utils.utils import *
from utils.graph import *
from src.model_batchdensegat import BatchDenseGAT, HeterEdgeDenseGAT
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
parser.add_argument('--tensorboard-log', type=str, default='debug', help="name of this run")
parser.add_argument('--dataset', type=str, default='Weibo-Aminer', help="available options are ['Weibo-Aminer','Twitter-Huangxin']")
# Weibo: subg_inf_5_400_deg_3_97_ego_49_neg_1_restart_0.2_1000
# Twitter: subg_inf_5_400_deg_3_62_ego_49_neg_1_restart_0.2_1000
parser.add_argument('--subgraph-dirname', type=str, default='subg_inf_5_400_deg_3_97_ego_49_neg_1_restart_0.2_1000', help="")
parser.add_argument('--model', type=str, default='heteredgedensegat', help="available options are ['batchdensegat','heteredgedensegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
parser.add_argument('--instance-normalization', action='store_true', default=False, help="Enable instance normalization")
parser.add_argument('--fine-tune', action='store_true', default=False, help="Fine Tune Model Global Embeddings")
parser.add_argument('--use-motif', action='store_true', default=False, help="Use Motif-Enhanced Graph")
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=1024, help="Batch size")
# [default] hetersparsegat: 3e-3, hypergat: 3e-3
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
# parser.add_argument('--unified-dim', type=int, default=64, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
# parser.add_argument('--patience', type=int, default=5, help='Patience for EarlyStopping')
parser.add_argument('--n-component', type=int, default=3, help="Number of Prominent Component Topic Classes Foreach Topic")
parser.add_argument('--window-size', type=int, default=200, help="Window Size of Building Topical Edges")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--train-ratio', type=float, default=75, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=12.5, help="Validation ratio (0, 100)")
parser.add_argument('--gpu', type=str, default="cuda:8", help="Select GPU")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/basic/graph.pkl")) # Total 44896 User Nodes
try:
    user_nodes = graph.vs["label"]
except:
    user_nodes = graph.vs.indices
n_user = graph.vcount()

feature_filename_suffix = f"_user{n_user}" if args.dataset == 'Weibo-Aminer' else ""
structural_feat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/vertex_feature{feature_filename_suffix}.npy"))
struc_cascade_feats = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/three_sort_feature{feature_filename_suffix}.npy"))
deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/deepwalk_emb{feature_filename_suffix}.data"), max_idx=user_nodes[-1]+1)

# fn = lambda val,ind: val if len(val)==len(ind) else val[ind]
# structural_feat = fn(structural_feat, ind=user_nodes)
structural_feat = preprocessing.scale(structural_feat)
# struc_cascade_feats = fn(struc_cascade_feats, ind=user_nodes)
struc_cascade_feats = preprocessing.scale(struc_cascade_feats)
# deepwalk_feat = fn(deepwalk_feat, ind=user_nodes)

# global_embs = [torch.FloatTensor(structural_feat), torch.FloatTensor(deepwalk_feat), torch.FloatTensor(struc_cascade_feats)]
global_embs = [torch.FloatTensor(structural_feat), torch.FloatTensor(deepwalk_feat),]
# n_feat = structural_feat.shape[1]+struc_cascade_feats.shape[1]+deepwalk_feat.shape[1]
n_feat = structural_feat.shape[1]+deepwalk_feat.shape[1]

tensorboard_log_dir = 'tensorboard/tensorboard_%s_subgraph_%s_model_%s_epochs%d_lr%f_batch%d' % (args.tensorboard_log, args.subgraph_dirname, args.model, args.epochs, args.lr, args.batch_size,)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
writer = SummaryWriter(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

n_units = [int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]

if args.model == 'batchdensegat':
    influence_dataset = InfluenceDataSet(
        file_dir=os.path.join(DATA_ROOTPATH, f"{args.dataset}/subgraph_sampling/{args.subgraph_dirname}"),
        seed=args.seed, shuffle=args.shuffle, model='gat')
    nb_classes = influence_dataset.get_num_class()
    n_feat = n_feat+influence_dataset.get_influence_feature_dimension()
    n_units = n_units + [nb_classes]
    n_heads = n_heads + [1]
    model = BatchDenseGAT(global_embs=global_embs, n_feat=n_feat, n_units=n_units, n_heads=n_heads,
        # attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, norm_mask=[0,1,0], fine_tune=args.fine_tune)
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, norm_mask=[0,1], fine_tune=args.fine_tune)

elif args.model == 'heteredgedensegat':
    base_filename = "topic_diffusion_graph" if not args.use_motif else "topic_diffusion_motif_graph"
    classid2simmat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/{base_filename}_windowsize{args.window_size}.data")) # type=='pyg.Data'

    def from_pygdata_to_igraph(pyg, num_user):
        ig = igraph.Graph(num_user, directed=True)
        ig.add_edges(list(zip(*pyg.edge_index.numpy())))
        ig.simplify()
        return ig
    hedge_graphs = [from_pygdata_to_igraph(hgraph, n_user) for _, hgraph in classid2simmat.items()]

    influence_dataset = InfluenceDataSet(
        file_dir=os.path.join(DATA_ROOTPATH, f"{args.dataset}/subgraph_sampling/{args.subgraph_dirname}"),
        seed=args.seed, shuffle=args.shuffle, model='gat', hedge_graphs=hedge_graphs)
    nb_classes = influence_dataset.get_num_class()
    n_feat = n_feat+influence_dataset.get_influence_feature_dimension()
    n_units = n_units + [nb_classes]
    n_heads = n_heads + [1]
    model = HeterEdgeDenseGAT(global_embs=global_embs, n_feat=n_feat, n_adj=len(hedge_graphs), n_units=n_units, n_heads=n_heads, 
        attn_dropout=args.attn_dropout, dropout=args.dropout, instance_normalization=args.instance_normalization, norm_mask=[0,1], fine_tune=args.fine_tune)
logger.info(f"n_user={n_user}, n_feat={n_feat}, n_units={n_units}, n_heads={n_heads}, shape_ret={(n_user,nb_classes)}")

class_weight = influence_dataset.get_class_weight()
N = len(influence_dataset)
train_start,  valid_start, test_start = 0, int(N*args.train_ratio/100), int(N*(args.train_ratio+args.valid_ratio)/100)
train_loader = DataLoader(influence_dataset, batch_size=args.batch_size, sampler=ChunkSampler(valid_start-train_start, 0))
valid_loader = DataLoader(influence_dataset, batch_size=args.batch_size, sampler=ChunkSampler(test_start-valid_start, valid_start))
test_loader = DataLoader(influence_dataset, batch_size=args.batch_size, sampler=ChunkSampler(N-test_start, test_start))
logger.info(f"train_start={train_start}, valid_start={valid_start}, test_start={test_start}, train_loader={len(train_loader)}, valid_loader={len(valid_loader)}, test_loader={len(test_loader)}")

if args.cuda:
    model.to(args.gpu)
    class_weight = class_weight.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# stopper = EarlyStopping(patience=args.patience)

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
        elif args.model == 'heteredgedensegat':
            output = model(graphs, vertices, influence_feats)
            output = output[:,-1,:]
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
        elif args.model == 'heteredgedensegat':
            output = model(graphs, vertices, influence_feats)
            output = output[:,-1,:]
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
