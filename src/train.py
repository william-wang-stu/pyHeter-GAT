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
from lib.log import logger
from utils import load_pickle, save_pickle, init_args, ChunkSampler, HeterGraphDataset, sparse_batch_collate
from model import BatchGAT, HeterGraphAttentionNetwork
import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from tensorboard_logger import tensorboard_logger

from torch.utils.data import Dataset
from utils import SubGraphSample, load_w2v_feature

class DiggDataset(Dataset):
    def __init__(self, samples: SubGraphSample, embedding) -> None:
        super().__init__()
        self.adjs = samples.adj_matrices
        self.labels = samples.labels
        self.feats = samples.influence_features
        self.vertex_ids = samples.vertex_ids
        self.concact_feats(embedding)
    def concact_feats(self, embedding):
        feats = []
        for idx, vertex_ids in enumerate(self.vertex_ids):
            emb_feats = [embedding[user] for user in vertex_ids]
            feats.append(np.concatenate((self.feats[idx], emb_feats), axis=1))
        self.feats = np.array(feats)
        logger.info(self.feats.shape)
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, index):
        return self.adjs[index], self.vertex_ids[index], self.labels[index], self.feats[index]

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
    return adjs_batch, vertices_batch, labels_batch, feats_batch

def digg_load_dataset(train_ratio=60, valid_ratio=20, batch_size=256):
    embedding_path = "/root/Lab_Related/data/Heter-GAT/Classic/deepwalk/deepwalk_added.emb_64"
    vertices = np.load("/root/TR-pptusn/DeepInf-preprocess/preprocess/stages_op_inf_100_1k/vertex_id.npy")
    max_vertex_idx = np.max(vertices)
    embedding = load_w2v_feature(embedding_path, max_vertex_idx)
    # embedding = torch.FloatTensor(embedding)

    samples = SubGraphSample(
        adj_matrices=np.load("/root/TR-pptusn/DeepInf-preprocess/preprocess/stages_op_inf_100_1k/adjacency_matrix.npy"),
        influence_features=np.load("/root/TR-pptusn/DeepInf-preprocess/preprocess/stages_op_inf_100_1k/influence_feature.npy"),
        vertex_ids=np.load("/root/TR-pptusn/DeepInf-preprocess/preprocess/stages_op_inf_100_1k/vertex_id.npy"),
        labels=np.load("/root/TR-pptusn/DeepInf-preprocess/preprocess/stages_op_inf_100_1k/label.npy")
    )
    dataset = DiggDataset(samples, embedding)
    nb_samples    = len(dataset)
    
    train_start,  valid_start, test_start = 0, int(nb_samples*train_ratio/100), int(nb_samples*(train_ratio+valid_ratio)/100)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(valid_start-train_start, 0), collate_fn=collate_fn2)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(test_start-valid_start, valid_start), collate_fn=collate_fn2)
    test_loader  = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(nb_samples - test_start, test_start), collate_fn=collate_fn2)
    logger.info(f"Finish Loading Dataset... train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")

    return samples, embedding, train_loader, valid_loader, test_loader

# 1. 
def load_dataset(data_filepath:str, train_ratio:float, valid_ratio:float, batch_size:int):
    # heter_samples = load_pickle(os.path.join(data_dirpath, "heter_samples_tensor.p"))
    heter_samples = load_pickle(data_filepath)
    dataset       = HeterGraphDataset(heter_samples=heter_samples)
    nb_samples    = len(dataset)
    
    train_start,  valid_start, test_start = 0, int(nb_samples*train_ratio/100), int(nb_samples*(train_ratio+valid_ratio)/100)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(valid_start-train_start, 0), collate_fn=sparse_batch_collate)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(test_start-valid_start, valid_start), collate_fn=sparse_batch_collate)
    test_loader  = DataLoader(dataset, batch_size=batch_size, sampler=ChunkSampler(nb_samples - test_start, test_start), collate_fn=sparse_batch_collate)
    logger.info(f"Finish Loading Dataset... train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")

    return heter_samples, train_loader, valid_loader, test_loader

args = init_args()
GPU_MODEL = args.gpu
# heter_samples, train_loader, valid_loader, test_loader = load_dataset(args.file_dir, args.train_ratio, args.valid_ratio, args.batch)
heter_samples, embedding, train_loader, valid_loader, test_loader = digg_load_dataset()
nb_samples = len(heter_samples)
nb_classes = 2
class_weight = torch.FloatTensor(nb_samples / (nb_classes*np.bincount(heter_samples.labels))) if args.class_weight_balanced else torch.ones(nb_classes)
nb_user = 50
# n_units = [heter_samples.initial_features.shape[2]]+[int(x) for x in args.hidden_units.strip().split(",")]
n_units = [heter_samples.influence_features.shape[2]+64]+[int(x) for x in args.hidden_units.strip().split(",")]
n_heads = [int(x) for x in args.heads.strip().split(",")]

# 2. 
# model = HeterGraphAttentionNetwork(n_user=nb_user, n_units=n_units, nb_classes=nb_classes, n_heads=n_heads, dropout=args.dropout)
model = BatchGAT(pretrained_emb=torch.FloatTensor(embedding), n_units=n_units, n_heads=n_heads, dropout=args.dropout)
if args.cuda:
    # model.to(GPU_MODEL)
    model.to(GPU_MODEL)
    # class_weight = class_weight.to(GPU_MODEL)
    class_weight = class_weight.to(GPU_MODEL)
# params = [{'params': model.parameters()}]
optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        # uu_adjs, ut_adjs, labels, feats = batch
        # # NOTE: https://stackoverflow.com/questions/69742930/runtimeerror-nll-loss-forward-reduce-cuda-kernel-2d-index-not-implemented-for
        # bs = uu_adjs.size(0)
        # if args.cuda:
        #     uu_adjs, ut_adjs, labels, feats = uu_adjs.to(GPU_MODEL), ut_adjs.to(GPU_MODEL), labels.to(GPU_MODEL), feats.to(GPU_MODEL)

        adjs, vertices, labels, feats = batch
        bs = adjs.size(0)
        if args.cuda:
            adjs, vertices, labels, feats = adjs.to(GPU_MODEL), vertices.to(GPU_MODEL), labels.to(GPU_MODEL), feats.to(GPU_MODEL)

        # output = model(feats, torch.stack([uu_adjs, ut_adjs]))
        # output = model(feats, torch.stack([adjs, adjs]))
        output = model(feats, vertices, adjs)
        output = output[:,-1,:] # choose last user

        loss_batch = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_batch.item()

        y_true += labels.data.tolist()
        # 返回output中每行最大值的索引
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
        # uu_adjs, ut_adjs, labels, feats = batch
        # bs = uu_adjs.size(0)
        # if args.cuda:
        #     uu_adjs, ut_adjs, labels, feats = uu_adjs.to(GPU_MODEL), ut_adjs.to(GPU_MODEL), labels.to(GPU_MODEL), feats.to(GPU_MODEL)

        adjs, vertices, labels, feats = batch
        bs = adjs.size(0)
        if args.cuda:
            adjs, vertices, labels, feats = adjs.to(GPU_MODEL), vertices.to(GPU_MODEL), labels.to(GPU_MODEL), feats.to(GPU_MODEL)

        optimizer.zero_grad()
        # output = model(torch.rand((feats.shape)).to(GPU_MODEL), torch.stack([uu_adjs, ut_adjs]))
        # output = model(feats, torch.stack([uu_adjs, ut_adjs]))
        # output = model(feats, torch.stack([adjs, adjs]))
        output = model(feats, vertices, adjs)
        output = output[:,-1,:] # choose last user

        loss_train = F.nll_loss(output, labels, class_weight)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss in this epoch %f", loss / total)
    # logger.info(f"GPU Mem Usage: {torch.cuda.memory_reserved(int(GPU_MODEL[-1]))/1024**3}")
    tensorboard_logger.log_value('train_loss', loss / total, epoch + 1)
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint!", epoch)
        best_thr = evaluate(epoch, valid_loader, return_best_thr=True, log_desc='valid_')
        evaluate(epoch, test_loader, thr=best_thr, log_desc='test_')

# 3. 
# Training
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
