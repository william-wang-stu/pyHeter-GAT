# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

from lib.utils import get_sparse_tensor
from lib.log import logger
from utils.utils import *
from utils.graph import *
from utils.tweet_clustering import tweet_centralized_process
from utils.metric import compute_metrics
from utils.Constants import PAD
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

def get_performance(crit, pred, gold):
    ''' Apply label smoothing if needed '''
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    # print ("get performance, ", gold.shape, pred.shape)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(PAD).data).sum().float()
    return loss, n_correct

def evaluate(epoch, best_thrs=None, return_best_thr=False, log_desc='valid_'):
    model.eval()

    loss, correct, total, prec, rec, f1 = 0., 0., 0., 0., 0., 0.
    y_true, y_pred, y_score, thrs = [], [], [], {}
    series_y_true, series_y_prob = [], []
    for hashtag in sorted(preprocess_timelines.keys()):
        if hashtag not in labels_mp or (args.model == 'heteredgegat' and hashtag not in tagid2classids): continue
        cascade_users = [elem[0] for elem in preprocess_timelines[hashtag]]

        labels, val_mask, test_mask, class_weight = labels_mp[hashtag], val_mask_mp[hashtag], test_mask_mp[hashtag], class_weight_mp[hashtag]
        if args.cuda:
            labels = labels.to(args.gpu); class_weight = class_weight.to(args.gpu)
        
        if args.model == 'densegat':
            output = model(adj, static_emb, dynamic_embs)
        elif args.model == 'heteredgegat':
            if not args.use_random_multiedge:
                hedge_adjs = [classid2simmat[classid].to(args.gpu) if args.cuda else classid2simmat[classid] for classid in tagid2classids[hashtag]]
                hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
            else:
                hedge_adjs = [random_classid2simmat[classid].to(args.gpu) if args.cuda else random_classid2simmat[classid] for classid in tagid2classids[hashtag]]
                hedge_adjs = hedge_adjs + [adj]*(args.n_component+1-len(hedge_adjs))
                # hedge_adjs = [adj] * (args.n_component+1)
            output = model(hedge_adjs, static_emb, dynamic_embs)
        
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
        correct += alpha * np.mean(np.array(y_true_cur) == np.array(y_pred_cur))
        loss  += alpha * loss_batch.item()
        total += alpha
        series_y_true.append([cascade_users[-1]])
        series_y_prob.append(y_score_cur)
    
    model.train()

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("%sloss: %.4f Acc: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f", log_desc, loss / total, correct / total, auc, prec, rec, f1)
    
    scores = compute_metrics(series_y_prob, series_y_true)
    logger.info(f"MRR={scores['MRR']}, hits@10={scores['hits@10']}, map@10={scores['map@10']}, hits@50={scores['hits@50']}, map@100={scores['map@100']}, hits@100={scores['hits@100']}, map@100={scores['map@100']},")
    
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'acc', correct / total, epoch+1)
    writer.add_scalar(log_desc+'mrr', scores['MRR'], epoch+1)
    writer.add_scalar(log_desc+'hits@10', scores['hits@10'], epoch+1)
    writer.add_scalar(log_desc+'map@10', scores['map@10'], epoch+1)
    writer.add_scalar(log_desc+'hits@50', scores['hits@50'], epoch+1)
    writer.add_scalar(log_desc+'map@50', scores['map@50'], epoch+1)
    writer.add_scalar(log_desc+'hits@100', scores['hits@100'], epoch+1)
    writer.add_scalar(log_desc+'map@100', scores['map@100'], epoch+1)
    writer.add_scalar(log_desc+'auc', auc, epoch+1)
    writer.add_scalar(log_desc+'prec', prec, epoch+1)
    writer.add_scalar(log_desc+'rec', rec, epoch+1)
    writer.add_scalar(log_desc+'f1', f1, epoch+1)

    if return_best_thr:
        return thrs
    else:
        return None

def train(epoch_i, batch_data, model, optimizer, loss_func, writer, log_desc='train_'):
    model.train()

    loss, correct, total = 0., 0., 0.
    for batch_i, batch in enumerate(tqdm(batch_data)):
        # if hashtag not in labels_mp or (args.model == 'heteredgegat' and hashtag not in tagid2classids): continue
        # labels, train_mask, class_weight = labels_mp[hashtag], train_mask_mp[hashtag], class_weight_mp[hashtag]
        # if args.cuda:
        #     labels = labels.to(args.gpu); class_weight = class_weight.to(args.gpu)

        cascade_users, cascade_tss, cascade_intervals, cascade_contents = (item.to(args.gpu) for item in batch)
        gold = cascade_users[:, 1:]

        optimizer.zero_grad()
        if args.model == 'densegat':
            output = model(adj, static_emb, dynamic_embs)
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
        loss, n_correct = get_performance(loss_func, output, gold)
        loss.backward()
        optimizer.step()
        optimizer.update_learning_rate()

        n_words = gold.data.ne(PAD).sum().float()
        loss += n_words * loss.item()
        correct += n_correct.item()
        total += n_words
    logger.info(f"epoch {epoch_i} loss={loss/total}, acc={n_correct/total}, GPU Memory Usage={check_gpu_memory_usage(int(args.gpu[-1]))} MiB")
    # tensorboard_logger.log_value(log_desc+'loss', loss / total, epoch+1)
    writer.add_scalar(log_desc+'loss', loss / total, epoch+1)

def train_epoch(model, training_data, graph, diffusion_graph, loss_func, optimizer, epoch_i):
    for i, batch in enumerate(tqdm(training_data)): # tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        tgt, tgt_timestamp = (item.cuda() for item in batch)

        start_time = time.time() 
        np.set_printoptions(threshold=np.inf)
        gold = tgt[:, 1:]

        n_words = gold.data.ne(PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        optimizer.zero_grad()
        pred = model(tgt, tgt_timestamp, graph, diffusion_graph)
        # backward
        loss, n_correct = get_performance(loss_func, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct.item()
        total_loss += loss.item()
        print("epoch " + str(epoch_i+1) + " Training batch ", i, " loss: ", loss.item(), " acc:", (n_correct.item()/len(pred)) )
        # print ("A Batch Time: ", str(time.time()-start_time))

    return total_loss/n_total_words, n_total_correct/n_total_words

def test_epoch(model, validation_data, graph, diffusion_graph, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    for i, batch in enumerate(validation_data):  #tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
        print("Validation batch ", i)
        # prepare data
        tgt, tgt_timestamp = batch
        y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

        # forward
        pred = model(tgt, tgt_timestamp, graph, diffusion_graph)
        y_pred = pred.detach().cpu().numpy()

        scores_batch, scores_len = metric.compute_metric(y_pred, y_gold, k_list)
        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores

def main():
    train_data = DataConstruct(dataset_dirpath="", batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, data_type=0, load_dict=True)
    valid_data = DataConstruct(dataset_dirpath="", batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, data_type=1, load_dict=False)
    test_data  = DataConstruct(dataset_dirpath="", batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, data_type=2, load_dict=False)

    loss_func = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=PAD)

    # optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = ScheduledOptim(
        optim.Adagrad(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 
        args.d_model, args.n_warmup_steps,)
    
    t_total = time.time()
    logger.info("training...")
    for epoch_i in range(args.epochs):
        start = time.time()
        train_loss, train_accu = train_epoch(model, train_data, diffusion_graph, loss_func, optimizer, epoch_i)
        logger.info('   - (Training)    loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, elapse=(time.time()-start)/60))
        
        if (epoch_i + 1) % args.check_point == 0:
            logger.info("epoch %d, checkpoint!", epoch)
            start = time.time()
            scores = test_epoch(model, valid_data, diffusion_graph)
            logger.info('   - (Validating)    scores: {scores}, elapse: {elapse:3.3f} min'.format(
                scores="/".join([f"{key}-{value}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60))
            
            scores = test_epoch(model, test_data, diffusion_graph)
            logger.info('   - (Testing)    scores: {scores}'.format(
                scores="/".join([f"{key}-{value}" for key,value in scores.items()]),))
    logger.info("Total Elapse: {elapse:3.3f} min".format(elapse=(time.time()-t_total)/60))

main()
