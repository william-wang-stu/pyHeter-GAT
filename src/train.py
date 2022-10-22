"""
总体思路: 
1. 输入数据: 异质图网络, 异质图节点特征
-   使用子网训练测试, 每个子网Sample上仅包括User节点, 这里为子图中的每个节点随机选取若干(N=20)个Tweet节点一并用作测试
    P.S. 注意此时子网大小为50Users+50*30Tweets=1550
2. 网路
3. 训练测试
"""
from random import sample
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))
from lib.log import logger
from utils import load_pickle, save_pickle, init_args, ChunkSampler, HeterGraphDataset, sparse_batch_collate
from model import BatchGAT, BatchGAT2, HeterdenseGAT
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from tensorboard_logger import tensorboard_logger
import shutil
import argparse

from torch.utils.data import Dataset
from utils import SubGraphSample, load_w2v_feature
import sklearn
from sklearn import preprocessing

class DiggDataset(Dataset):
    def __init__(self, samples: SubGraphSample, embedding, args) -> None:
        super().__init__()
        self.adjs = samples.adj_matrices
        self.labels = samples.labels
        self.feats = samples.influence_features
        self.vertex_ids = samples.vertex_ids
        self.set_dtype()
        if not args.use_pretrained_emb:
            self.concact_feats(embedding)
        self.extend_graph()
    def extend_graph(self):
        # self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.adjs.shape[1])
        self.adjs += identity
        self.adjs[self.adjs != 0] = 1.0
        self.adjs = self.adjs.astype(np.dtype('B'))
    def concact_feats(self, embedding):
        feats = []
        for idx, vertex_ids in enumerate(self.vertex_ids):
            emb_feats = [embedding[user] for user in vertex_ids]
            feats.append(np.concatenate((self.feats[idx], emb_feats), axis=1))
        self.feats = np.array(feats)
    def set_dtype(self):
        self.adjs  = np.float64(self.adjs)
        self.feats = np.float32(self.feats)
    def get_user_num(self):
        return self.feats.shape[1]
    def get_feature_dim(self):
        return self.feats.shape[2]
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, index):
        return self.adjs[index], self.feats[index], self.labels[index], self.vertex_ids[index]

def collate_fn2(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    adjs_batch, vertices_batch, labels_batch, feats_batch = zip(*batch)
    adjs_batch = torch.FloatTensor(np.array(adjs_batch))
    vertices_batch = torch.LongTensor(np.array(vertices_batch))
    
    if type(labels_batch[0]).__module__ == 'numpy':
        # NOTE: https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        labels_batch = torch.LongTensor(labels_batch)
    
    if type(feats_batch[0]).__module__ == 'numpy':
        feats_batch = torch.FloatTensor(np.array(feats_batch))
    return adjs_batch, feats_batch, labels_batch, vertices_batch

def digg_load_dataset(args, train_ratio=60, valid_ratio=20, batch_size=256):
    data_dirpath = args.file_dir
    # embedding_path = "/root/data/TR/DeepInf-4dataset/digg/deepwalk.emb_64"
    embedding_path = "/root/data/HeterGAT/basic/deepwalk/deepwalk_added.emb_64"
    vertices = np.load(os.path.join(data_dirpath, "vertex_id.npy"))
    max_vertex_idx = np.max(vertices)
    embedding = load_w2v_feature(embedding_path, max_vertex_idx)

    samples = SubGraphSample(
        adj_matrices=np.load(os.path.join(data_dirpath, "adjacency_matrix.npy")),
        influence_features=np.load(os.path.join(data_dirpath, "influence_feature.npy")),
        vertex_ids=np.load(os.path.join(data_dirpath, "vertex_id.npy")),
        labels=np.load(os.path.join(data_dirpath, "label.npy"))
    )
    dataset = DiggDataset(samples, embedding, args)
    nb_classes = 2
    class_weight = torch.FloatTensor(len(dataset) / (nb_classes*np.bincount(samples.labels))) if args.class_weight_balanced else torch.ones(nb_classes)
    feature_dim = dataset.get_feature_dim()
    
    return dataset, embedding, class_weight, feature_dim, nb_classes

args = init_args()
GPU_MODEL = args.gpu
dataset, embedding, class_weight, feature_dim, nb_classes = digg_load_dataset(args)
N = len(dataset)
logger.info(f"len={N}")
n_units = [feature_dim]+[int(x) for x in args.hidden_units.strip().split(",")]+[nb_classes]
n_heads = [int(x) for x in args.heads.strip().split(",")]+[1]
logger.info("class_weight=%.2f:%.2f", class_weight[0], class_weight[1])
logger.info("feature dimension=%d", feature_dim)
logger.info("number of classes=%d", nb_classes)

train_start,  valid_start, test_start = 0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
train_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N - test_start, test_start))
logger.info(f"Finish Loading Dataset... train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")

# model = HeterGraphAttentionNetwork(n_user=50, n_units=n_units, n_heads=n_heads, dropout=args.dropout)
if args.use_pretrained_emb:
    model = HeterdenseGAT(n_user=dataset.get_user_num(), pretrained_emb=torch.FloatTensor(embedding), nb_node_kinds=2, nb_classes=nb_classes, n_units=n_units, n_heads=n_heads, dropout=args.dropout)
else:
    model = BatchGAT2(n_units=n_units, n_heads=n_heads, dropout=args.dropout)

if args.cuda:
    model.to(args.gpu)
    class_weight = class_weight.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.to(args.gpu)
            graph = graph.to(args.gpu)
            labels = labels.to(args.gpu)
            vertices = vertices.to(args.gpu)
        if not args.use_pretrained_emb:
            output = model(features, graph)
        else:
            output = model(features, vertices, [graph, graph])
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += bs

    model.train()

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
            log_desc, loss / total, auc, prec, rec, f1)

    tensorboard_logger.log_value(log_desc + 'loss', loss / total, epoch + 1)
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


def train(epoch, train_loader, valid_loader, test_loader, log_desc='train_'):
    model.train()

    loss = 0.
    total = 0.
    for i_batch, batch in enumerate(train_loader):
        graph, features, labels, vertices = batch
        bs = graph.size(0)

        if args.cuda:
            features = features.to(args.gpu)
            graph = graph.to(args.gpu)
            labels = labels.to(args.gpu)
            vertices = vertices.to(args.gpu)

        optimizer.zero_grad()
        if not args.use_pretrained_emb:
            output = model(features, graph)
        else:
            output = model(features, vertices, [graph, graph])
        if args.model == "gcn" or args.model == "gat":
            output = output[:, -1, :]
        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    tensorboard_logger.log_value('train_loss', loss / total, epoch + 1)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')


# Train model
t_total = time.time()
logger.info("training...")
for epoch in range(args.epochs):
    train(epoch, train_loader, valid_loader, test_loader)
logger.info("optimization Finished!")
logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

logger.info("retrieve best threshold...")
best_thr = evaluate(args.epochs, valid_loader, return_best_thr=True, log_desc='valid_')

# Testing
logger.info("testing...")
evaluate(args.epochs, test_loader, thr=best_thr, log_desc='test_')
