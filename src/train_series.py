# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import get_sparse_tensor
from lib.log import logger
from utils.utils import *
from utils.graph import *
from utils.graph_aminer import *
from utils.tweet_clustering import tweet_centralized_process
from utils.metric import compute_metrics
from utils.Constants import PAD, EOS
from utils.Optim import ScheduledOptim
from src.data_loader import DataConstruct
from src.model import DenseSparseGAT, HeterEdgeSparseGAT
import numpy as np
import argparse
import shutil
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from torch_geometric.data import Data
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--model', type=str, default='densegat', help="available options are ['densegat','heteredgegat','hetersparsegat','hypergat','hypergatwithhetersparsegat']")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--class-weight-balanced', action='store_true', default=True, help="Adjust weights inversely proportional to class frequencies in the input data")
# >> Preprocess
parser.add_argument('--min-user-participate', type=int, default=2, help="Min User Participate in One Cascade")
parser.add_argument('--max-user-participate', type=int, default=500, help="Max User Participate in One Cascade")
parser.add_argument('--train-ratio', type=float, default=0.8, help="Training ratio (0, 1)")
parser.add_argument('--valid-ratio', type=float, default=0.1, help="Validation ratio (0, 1)")
# >> Model
parser.add_argument('--tmax', type=int, default=120, help="Max Time in the Observation Window")
parser.add_argument('--n-interval', type=int, default=40, help="Number of Time Intervals in the Observation Window")
parser.add_argument('--n-component', type=int, default=3, help="Number of Prominent Component Topic Classes Foreach Topic")
parser.add_argument('--window-size', type=int, default=200, help="Window Size of Building Topical Edges")
parser.add_argument('--instance-normalization', action='store_true', default=False, help="Enable instance normalization")
parser.add_argument('--use-tweet-feat', action='store_true', default=False, help="Use Tweet-Side Feat Aggregated From Tag Embeddings")
parser.add_argument('--unified-dim', type=int, default=128, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--d_model', type=int, default=64, help='Options in ScheduledOptim')
parser.add_argument('--n_warmup_steps', type=int, default=1000, help='Options in ScheduledOptim')
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, help='Number of epochs to train.')
# [default] hetersparsegat: 3e-3, hypergat: 3e-3, densegat: 3e-2
parser.add_argument('--lr', type=float, default=3e-2, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="16,16", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="8,8", help="Heads in each layer, splitted with comma")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--gpu', type=str, default="cuda:1", help="Select GPU")
# >> Ablation Study
parser.add_argument('--use-random-multiedge', action='store_true', default=False, help="Use Random Multi-Edge to build Heter-Edge-Matrix if set true (Available only when model='heteredgegat')")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
logger.info(f"Args: {args}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def get_cascade_criterion(user_size):
    ''' With PAD token zero weight '''
    weight = torch.ones(user_size)
    weight[PAD] = 0
    weight[EOS] = 0
    return torch.nn.CrossEntropyLoss(weight, size_average=False)

def get_performance(criterion, pred_cascade, gold_cascade):
    '''
    pred_cascade: (#samples, #user_size), gold_cascade: (#samples,)
    '''
    pred_cascade = pred_cascade.max(1)[1]
    gold_cascade = gold_cascade.contiguous().view(-1)
    loss = criterion(pred_cascade, gold_cascade)

    n_correct = pred_cascade.data.eq(gold_cascade.data)
    n_correct = n_correct.masked_select(gold_cascade.ne(PAD).data).sum().float()
    # n_correct = n_correct.masked_select((gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data).sum().float()
    return loss, n_correct

def get_scores(pred_cascade, gold_cascade, k_list=[10,50,100]):
    # pred_cascade = pred_cascade.max(1)[0].detach().cpu().numpy()
    pred_cascade = pred_cascade.detach().cpu().numpy()
    gold_cascade = gold_cascade.contiguous().view(-1).detach().cpu().numpy()
    scores = compute_metrics(pred_cascade, gold_cascade, k_list)
    return scores

def train(epoch_i, batch_data, model, optimizer, loss_func, writer, log_desc='train_'):
    model.train()

    loss, correct, total = 0., 0., 0.
    for _, batch in enumerate(tqdm(batch_data)):
        # if hashtag not in labels_mp or (args.model == 'heteredgegat' and hashtag not in tagid2classids): continue
        # labels, train_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], class_weight_mp[hashtag]
        # if args.cuda:
        #     labels = labels.to(args.gpu); class_weight = class_weight.to(args.gpu)

        cascade_users, cascade_tss, cascade_intervals, cascade_contents = (item.to(args.gpu) for item in batch)
        gold_cascade = cascade_users[:, 1:]

        optimizer.zero_grad()
        if args.model == 'densegat':
            pred_cascade = model(adj, static_emb, dynamic_embs)
        # elif args.model == 'heteredgegat':
        #     if not args.use_random_multiedge:
        #         hedge_adjs = [classid2simmat[classid].to(args.gpu) if args.cuda else classid2simmat[classid] for classid in tagid2classids[hashtag]]
        #         hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
        #     else:
        #         hedge_adjs = [random_classid2simmat[classid].to(args.gpu) if args.cuda else random_classid2simmat[classid] for classid in tagid2classids[hashtag]]
        #         hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
        #         # hedge_adjs = [adj] * (args.n_component+1)
        #     output = model(hedge_adjs, static_emb, dynamic_embs)
        # loss_train = F.nll_loss(output[train_mask], labels[train_mask], class_weight)
        # alpha = train_mask.sum().item()
        # loss  += alpha * loss_train.item()
        # total += alpha
        loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
        loss_batch.backward()
        optimizer.step()
        optimizer.update_learning_rate()

        n_words = gold_cascade.ne(PAD).data.sum().float()
        # n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
        loss += loss_batch.item()
        correct += n_correct.item()
        total += n_words
    logger.info(f"epoch {epoch_i} loss={loss/total}, acc={n_correct/total}, GPU Memory Usage={check_gpu_memory_usage(int(args.gpu[-1]))} MiB")
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1)
    writer.add_scalar(log_desc+'acc',  n_correct/total, epoch_i+1)

def evaluate(epoch_i, batch_data, model, optimizer, loss_func, writer, k_list=[10,50,100], log_desc='valid_'):
    model.eval()

    loss, correct, total = 0., 0., 0.
    scores = {'MRR': 0,}
    for k in k_list:
        scores[f'hits@{k}'] = 0
        scores[f'map@{k}'] = 0
    # loss, correct, total, prec, rec, f1 = 0., 0., 0., 0., 0., 0.
    # y_true, y_pred, y_score, series_y_true, series_y_prob = [], [], [], [], []
    for _, batch in enumerate(tqdm(batch_data)):
        # if hashtag not in labels_mp or (args.model == 'heteredgegat' and hashtag not in tagid2classids): continue
        # cascade_users = [elem[0] for elem in preprocess_timelines[hashtag]]

        cascade_users, cascade_tss, cascade_intervals, cascade_contents = (item.to(args.gpu) for item in batch)
        gold_cascade = cascade_users[:, 1:]

        # labels, train_mask, val_mask, test_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag], class_weight_mp[hashtag]
        # mask = train_mask + val_mask + test_mask
        # if args.cuda:
        #     labels = labels.to(args.gpu); class_weight = class_weight.to(args.gpu)
        
        if args.model == 'densegat':
            pred_cascade = model(adj, static_emb, dynamic_embs)
        # elif args.model == 'heteredgegat':
        #     if not args.use_random_multiedge:
        #         hedge_adjs = [classid2simmat[classid].to(args.gpu) if args.cuda else classid2simmat[classid] for classid in tagid2classids[hashtag]]
        #         hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
        #     else:
        #         hedge_adjs = [random_classid2simmat[classid].to(args.gpu) if args.cuda else random_classid2simmat[classid] for classid in tagid2classids[hashtag]]
        #         hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
        #         # hedge_adjs = [adj] * (args.n_component+1)
        #     output = model(hedge_adjs, static_emb, dynamic_embs)
        
        loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
        n_words = gold_cascade.ne(PAD).data.sum().float()
        # n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
        loss += loss_batch.item()
        correct += n_correct.item()
        total += n_words

        scores_batch = get_scores(pred_cascade, gold_cascade, k_list)
        scores['MRR'] += scores_batch['MRR'] * n_words
        for k in k_list:
            scores[f'hits@{k}'] += scores_batch[f'hits@{k}'] * n_words
            scores[f'map@{k}'] += scores_batch[f'map@{k}'] * n_words
    
    model.train()
    
    scores['MRR'] /= total
    for k in k_list:
        scores[f'hits@{k}'] /= total
        scores[f'map@{k}'] /= total
    logger.info(f"MRR={scores['MRR']}, hits@10={scores['hits@10']}, map@10={scores['map@10']}, hits@50={scores['hits@50']}, map@100={scores['map@100']}, hits@100={scores['hits@100']}, map@100={scores['map@100']},")
    
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1); writer.add_scalar(log_desc+'acc', correct/total, epoch_i+1); writer.add_scalar(log_desc+'mrr', scores['MRR'], epoch_i+1)
    writer.add_scalar(log_desc+'hits@10', scores['hits@10'], epoch_i+1);  writer.add_scalar(log_desc+'map@10', scores['map@10'], epoch_i+1)
    writer.add_scalar(log_desc+'hits@50', scores['hits@50'], epoch_i+1);  writer.add_scalar(log_desc+'map@50', scores['map@50'], epoch_i+1)
    writer.add_scalar(log_desc+'hits@100', scores['hits@100'], epoch_i+1);writer.add_scalar(log_desc+'map@100', scores['map@100'], epoch_i+1)
    # writer.add_scalar(log_desc+'auc', auc, epoch_i+1);                    writer.add_scalar(log_desc+'f1', f1, epoch_i+1)
    # writer.add_scalar(log_desc+'prec', prec, epoch_i+1);                  writer.add_scalar(log_desc+'rec', rec, epoch_i+1)

def main():
    # torch.set_num_threads(4)

    user_ids = read_user_ids()
    edges = get_static_subnetwork(user_ids)
    edges = list(zip(*edges))
    edges_t = torch.LongTensor(edges).t()
    weight_t = torch.FloatTensor([1]*len(edges))
    graph = Data(edge_index=edges_t, edge_weight=weight_t)

    train_data = DataConstruct(dataset_dirpath="", batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, data_type=0, load_dict=True)
    valid_data = DataConstruct(dataset_dirpath="", batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, data_type=1, load_dict=False)
    test_data  = DataConstruct(dataset_dirpath="", batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, data_type=2, load_dict=False)

    loss_func = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=PAD)
    # loss_func = get_cascade_criterion(train_data.user_size)

    # optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = ScheduledOptim(
        optim.Adagrad(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 
        args.d_model, args.n_warmup_steps,)
    
    if args.cuda:
        # model = model.to(args.gpu)
        loss_func = loss_func.to(args.gpu)
    
    tensorboard_log_dir = 'tensorboard/tensorboard_%s_stage%d_epochs%d_lr%f_t%d' % (args.tensorboard_log, args.stage, args.epochs, args.lr, args.tweet_per_user)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    shutil.rmtree(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

    t_total = time.time()
    logger.info("training...")
    for epoch_i in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train(epoch_i, train_data, model, optimizer, loss_func, writer)
        logger.info('   - (Training)    loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100*train_acc, elapse=(time.time()-start)/60))
        
        if (epoch_i + 1) % args.check_point == 0:
            logger.info("epoch %d, checkpoint!", epoch_i)
            start = time.time()
            scores = evaluate(epoch_i, valid_data, model, optimizer, loss_func, writer)
            logger.info('   - (Validating)    scores: {scores}, elapse: {elapse:3.3f} min'.format(
                scores="/".join([f"{key}-{value}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60))
            
            scores = evaluate(epoch_i, test_data, model, optimizer, loss_func, writer)
            logger.info('   - (Testing)    scores: {scores}'.format(
                scores="/".join([f"{key}-{value}" for key,value in scores.items()]),))
            save_model(epoch_i, args, model, optimizer)
    logger.info("Total Elapse: {elapse:3.3f} min".format(elapse=(time.time()-t_total)/60))

main()
