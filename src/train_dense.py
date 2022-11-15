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
from utils import load_pickle, save_pickle, ChunkSampler, SubGraphSample, load_w2v_feature, gen_random_tweet_ids, extend_subnetwork, DATA_ROOTPATH
from model import BatchdenseGAT, HeterdenseGAT
import numpy as np
import argparse
import shutil
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from tensorboard_logger import tensorboard_logger
from torch.utils.data import Dataset

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}")

# NOTE: Preliminaries: gen_random_tweet_ids(), extend_subnetwork() -> hs_stages_*/{extended_adjs.p, extended_vertices.p, sample_ids.p}
class HeterDataset(Dataset):
    def __init__(self, samples: SubGraphSample, args) -> None:
        super().__init__()
        self.adjs = samples.adj_matrices
        self.labels = samples.labels
        self.feats = samples.influence_features
        self.vertex_ids = samples.vertex_ids
        self.tags = samples.tags
        self.stages = samples.time_stages
        self.file_dir = args.file_dir
        self.user_num = samples.influence_features.shape[1]
        if args.model == 'heterdensegat':
            self.extend_hetergraph()
        self.add_self_loop()
        self.set_dtype()
    
    def extend_hetergraph(self):
        # NOTE: extend vertex_ids and adjs according to tweet_ids
        hs_filedir = os.path.join(DATA_ROOTPATH, self.file_dir).replace('stages_', 'hs_')
        self.adjs = load_pickle(os.path.join(hs_filedir, "extended_adjs.p"))
        vertex_ids = load_pickle(os.path.join(hs_filedir, "extended_vertices.p"))
        num_tweets = vertex_ids.shape[1] - self.vertex_ids.shape[1]
        self.vertex_ids = vertex_ids
        tweet_features = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/basic/doc2topic_tweetfeat.p"))
        ids = load_pickle(os.path.join(hs_filedir, "sample_ids.p"))
        
        user_feats, tweet_feats = self.feats[ids], tweet_features[self.vertex_ids[:,20:]]
        tmp1 = np.append(user_feats, np.zeros((user_feats.shape[0], num_tweets, user_feats.shape[2])), axis=1)
        tmp2 = np.append(np.zeros((tweet_feats.shape[0], self.user_num, tweet_feats.shape[2])), tweet_feats, axis=1)
        self.feats = np.concatenate((tmp1, tmp2), axis=2)
        self.labels, self.tags, self.stages = self.labels[ids], self.tags[ids], self.stages[ids]
        logger.info(f"{self.adjs.shape}, {self.vertex_ids.shape}, {self.labels.shape}, {self.feats.shape}, {self.tags.shape}, {self.stages.shape}")
    
    def add_self_loop(self):
        # NOTE: self-loop trick, the input graphs should have no self-loop
        identity = np.identity(self.adjs.shape[2], dtype='B')
        if args.model == 'batchdensegat':
            self.adjs += identity
        elif args.model == 'heterdensegat':
            for idx in range(self.adjs.shape[1]):
                self.adjs[:,idx] += identity
    
    def set_dtype(self):
        self.feats = np.float32(self.feats)
    
    def get_user_num(self):
        return self.user_num
    
    def get_feature_dim(self):
        return self.feats.shape[2]
    
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        return self.adjs[index], self.feats[index], self.labels[index], self.vertex_ids[index], self.tags[index], self.stages[index]

def gen_user_emb(tot_user_num):
    # user_feats = [[0.]*200*8*3 for _ in range(tot_user_num)]
    # for tag in range(200):
    #     logger.info(f"tag={tag}")
    #     for stage in range(8):
    #         for feats_idx, feats in enumerate(["norm_gravity_feature", "norm_exptime_feature1", "norm_ce_feature"]):
    #             feats = load_pickle(f"/root/data/HeterGAT/user_features/{feats}/hashtag{tag}_t{stage}.p")
    #             for idx in range(tot_user_num):
    #                 user_feats[idx][tag*3*8+stage*3+feats_idx] = float(feats[idx])
    # user_feats = np.array(user_feats)
    user_feats = load_pickle(os.path.join(DATA_ROOTPATH, "HeterGAT/user_features/user_features.p"))
    return torch.FloatTensor(user_feats)

def heter_load_dataset(args):
    data_dirpath = os.path.join(DATA_ROOTPATH, args.file_dir)
    embedding_path = os.path.join(DATA_ROOTPATH, "HeterGAT/basic/deepwalk/deepwalk_added.emb_64")
    vertices = np.load(os.path.join(data_dirpath, "vertex_id.npy"))
    max_vertex_idx = np.max(vertices)
    embedding = load_w2v_feature(embedding_path, max_vertex_idx)
    user_emb = gen_user_emb(max(max_vertex_idx, 208894))

    samples = SubGraphSample(
        adj_matrices=np.load(os.path.join(data_dirpath, "adjacency_matrix.npy")),
        influence_features=np.load(os.path.join(data_dirpath, "influence_feature.npy")),
        vertex_ids=np.load(os.path.join(data_dirpath, "vertex_id.npy")),
        labels=np.load(os.path.join(data_dirpath, "label.npy")),
        tags=np.load(os.path.join(data_dirpath, "hashtag.npy")),
        time_stages=np.load(os.path.join(data_dirpath, "stage.npy"))
    )
    dataset = HeterDataset(samples, args)
    nb_classes = 2
    class_weight = torch.FloatTensor(len(dataset) / (nb_classes*np.bincount(samples.labels))) if args.class_weight_balanced else torch.ones(nb_classes)
    feature_dim = dataset.get_feature_dim()
    
    return dataset, embedding, user_emb, class_weight, feature_dim, nb_classes

parser = argparse.ArgumentParser()
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='heterdensegat', help="available options are ['batchdensegat', 'heterdensegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--file-dir', type=str, required=True, help="Input file directory")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.5, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--batch', type=int, default=1024, help="Batch size")
parser.add_argument('--loop', type=int, default=1, help="Loop")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--train-ratio', type=float, default=60, help="Training ratio (0, 100)")
parser.add_argument('--valid-ratio', type=float, default=20, help="Validation ratio (0, 100)")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
parser.add_argument('--instance-normalization', action='store_true', default=True, help="Enable instance normalization")
parser.add_argument('--gpu', type=str, default="cuda:1", help="Select GPU")
# parser.add_argument('--use-infdataset', action='store_true', default=False, help="Use Influence Dataset")
# parser.add_argument('--use-pretrained-emb', action='store_true', default=True, help="Use Pretrained Emb, Else concact emb with other feats")
parser.add_argument('--use-user-emb', action='store_true', default=False, help="Whether to use User Emb")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

tensorboard_log_dir = 'tensorboard/%s_%s_lr%f_bs%d_hid%s:%s' % (args.model, args.tensorboard_log, args.lr, args.batch, args.hidden_units, args.heads)
os.makedirs(tensorboard_log_dir, exist_ok=True)
shutil.rmtree(tensorboard_log_dir)
tensorboard_logger.configure(tensorboard_log_dir)
logger.info('tensorboard logging to %s', tensorboard_log_dir)

dataset, embedding, user_emb, class_weight, feature_dim, nb_classes = heter_load_dataset(args)
N = len(dataset)
n_units = [feature_dim]+[int(x) for x in args.hidden_units.strip().split(",")]+[nb_classes]
n_heads = [int(x) for x in args.heads.strip().split(",")]+[1]
logger.info(f"Preparing Args... samples={N}, class_weight={class_weight[0]:.2f}:{class_weight[1]:.2f}, feature_dim={feature_dim}, nb_classes={nb_classes}, n_units={n_units}, n_heads={n_heads}")
logger.info(f"pretrained_emb size={embedding.shape}, user_emb size={user_emb.size()}")

train_start,  valid_start, test_start = 0, int(N * args.train_ratio / 100), int(N * (args.train_ratio + args.valid_ratio) / 100)
train_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(N - test_start, test_start))
logger.info(f"Loading Dataset... train={len(train_loader)}, valid={len(valid_loader)}, test={len(test_loader)}")

if args.model == 'batchdensegat':
    model = BatchdenseGAT(pretrained_emb=torch.FloatTensor(embedding), n_units=n_units, n_heads=n_heads, 
        attn_dropout=args.attn_dropout, dropout=args.dropout, use_user_emb=args.use_user_emb)
elif args.model == 'heterdensegat':
    model = HeterdenseGAT(n_user=dataset.get_user_num(), pretrained_emb=torch.FloatTensor(embedding), 
        nb_node_kinds=2, nb_classes=nb_classes, n_units=n_units, n_heads=n_heads, 
        attn_dropout=args.attn_dropout, dropout=args.dropout, use_user_emb=args.use_user_emb)
logger.info(f"model: {model}")

if args.cuda:
    user_emb = user_emb.to(args.gpu)
    model.to(args.gpu)
    class_weight = class_weight.to(args.gpu)

optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def evaluate(epoch, loader, thr=None, return_best_thr=False, log_desc='valid_'):
    model.eval()
    total = 0.
    loss, prec, rec, f1 = 0., 0., 0., 0.
    y_true, y_pred, y_score = [], [], []
    for i_batch, batch in enumerate(loader):
        for _ in range(args.loop):
            graph, features, labels, vertices, tags, stages = batch
            bs = graph.size(0)

            if args.cuda:
                features, graph, labels, vertices, tags, stages = features.to(args.gpu), graph.to(args.gpu), labels.to(args.gpu), vertices.to(args.gpu), tags.to(args.gpu), stages.to(args.gpu)
            
            if args.use_user_emb:
                user_feats = torch.stack([
                    user_emb[:,tags*8*3+stages*3+idx].T.gather(dim=1, index=vertices)
                for idx in range(3)], dim=2)
                user_feats = user_feats.to(args.gpu)
            else:
                user_feats = None
            
            if args.model == 'batchdensegat':
                output = model(vertices, graph, features, user_feats)
            elif args.model == 'heterdensegat':
                output = model(vertices, [graph[:,0], graph[:,1]], features, user_feats)
            if args.model[-3:] == "gat":
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
        for _ in range(args.loop):
            graph, features, labels, vertices, tags, stages = batch
            bs = graph.size(0)

            if args.cuda:
                features, graph, labels, vertices, tags, stages = features.to(args.gpu), graph.to(args.gpu), labels.to(args.gpu), vertices.to(args.gpu), tags.to(args.gpu), stages.to(args.gpu)
            
            optimizer.zero_grad()
            if args.use_user_emb:
                user_feats = torch.stack([
                    user_emb[:,tags*8*3+stages*3+idx].T.gather(dim=1, index=vertices)
                for idx in range(3)], dim=2)
                user_feats = user_feats.to(args.gpu)
            else:
                user_feats = None
            
            if args.model == 'batchdensegat':
                output = model(vertices, graph, features, user_feats)
            elif args.model == 'heterdensegat':
                output = model(vertices, [graph[:,0], graph[:,1]], features, user_feats)
            if args.model[-3:] == "gat":
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
