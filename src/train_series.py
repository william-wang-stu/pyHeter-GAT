# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))

import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from lib.log import logger
from utils.utils import *
from utils.graph import build_heteredge_mats
from utils.graph_aminer import *
from utils.metric import compute_metrics
from utils.Constants import PAD, EOS
from utils.Optim import ScheduledOptim
from utils.Patience import EarlyStopping
from src.data_loader import DataConstruct
from src.model_pyg import *
from src.sota.TAN.model import TAN
from src.sota.TAN.Option import Option
from src.sota.DHGPNTM.DyHGCN import DyHGCN_H
from src.sota.DHGPNTM.DataConstruct import LoadDynamicHeteGraph
from src.sota.FOREST.model import RNNModel
from src.sota.NDM.transformer.Models import Decoder
import numpy as np
import argparse
import shutil
import time
import torch
import torch.optim as optim
from torch_geometric.data import Data
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger.info(f"Reading From config.ini... DATA_ROOTPATH={DATA_ROOTPATH}, Ntimestage={Ntimestage}")

parser = argparse.ArgumentParser()
# >> Constant
parser.add_argument('--tensorboard-log', type=str, default='exp', help="name of this run")
parser.add_argument('--dataset', type=str, default='Weibo-Aminer', help="available options are ['Weibo-Aminer','Twitter-Huangxin']")
parser.add_argument('--model', type=str, default='heteredgegat', help="available options are ['densegat','heteredgegat','diffusiongat','dhgpntm']")
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
parser.add_argument('--use-gat', action='store_true', default=False, help="Use GAT as Backbone")
parser.add_argument('--use-topic-preference', action='store_true', default=False, help="Use Hand-crafted Topic Preference Weights to Aggregate topic-enhanced graph embeds")
parser.add_argument('--use-tweet-feat', action='store_true', default=False, help="Use Tweet-Side Feat Aggregated From Tag Embeddings")
parser.add_argument('--unified-dim', type=int, default=128, help='Unified Dimension of Different Feature Spaces.')
parser.add_argument('--d_model', type=int, default=64, help='Options in ScheduledOptim')
parser.add_argument('--n_warmup_steps', type=int, default=1000, help='Options in ScheduledOptim')
parser.add_argument('--patience', type=int, default=10, help='Patience Steps of EarlyStopping')
# >> Hyper-Param
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=32, help='Number of epochs to train.')
# [default] hetersparsegat: 3e-3, hypergat: 3e-3, densegat: 3e-2
parser.add_argument('--lr', type=float, default=3e-2, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--attn-dropout', type=float, default=0.0, help='Attn Dropout rate (1 - keep probability).')
parser.add_argument('--hidden-units', type=str, default="32,32", help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--heads', type=str, default="2,2", help="Heads in each layer, splitted with comma")
parser.add_argument('--check-point', type=int, default=10, help="Check point")
parser.add_argument('--gpu', type=str, default="cuda:6", help="Select GPU")
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
    return torch.nn.CrossEntropyLoss(weight)

def get_performance(criterion, pred_cascade, gold_cascade):
    '''
    pred_cascade: (#samples, #user_size), gold_cascade: (#samples,)
    '''
    gold_cascade = gold_cascade.contiguous().view(-1)
    pred_cascade[pred_cascade == -float('inf')] = 0
    pred_cascade = pred_cascade.view(gold_cascade.size(0), -1)
    loss = criterion(pred_cascade, gold_cascade)

    pred_cascade = pred_cascade.max(1)[1]
    n_correct = pred_cascade.data.eq(gold_cascade.data)
    n_correct = n_correct.masked_select((gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data).sum().float()
    return loss, n_correct

def get_performance2(opt, cascade_crit, pred_cascade, gold_cascade):  
    gold_cascade = gold_cascade.contiguous().view(-1)
    inte_sta = np.array([0]*opt.num_heads)
    pred_cascade = pred_cascade.view(gold_cascade.size(0),opt.num_heads,opt.user_size).max(1)[0]
    pred_cascade[pred_cascade == -float('inf')] = 0
    cascade_loss = cascade_crit(pred_cascade, gold_cascade)
    # gold_cate = torch.from_numpy(np.array(range(opt.num_heads))).unsqueeze(-1).repeat(1,opt.user_size).view(-1).to(opt.device)
    # regular_loss = regular_crit(regular_outputs,gold_cate)
    #cascade prediction
    pred_cascade = pred_cascade.max(1)[1]
    gold_cascade = gold_cascade.contiguous().view(-1)
    n_cascade_correct = pred_cascade.data.eq(gold_cascade.data)
    n_cascade_correct = n_cascade_correct.masked_select((gold_cascade.ne(PAD)*gold_cascade.ne(EOS)).data).sum().float()
    return cascade_loss, None, n_cascade_correct,inte_sta

def get_scores(pred_cascade:torch.Tensor, gold_cascade:torch.Tensor, k_list=[10,50,100]):
    gold_cascade = gold_cascade.contiguous().view(-1)           # (#samples,)
    pred_cascade = pred_cascade.view(gold_cascade.size(0), -1)  # (#samples, #user_size)
    pred_cascade = pred_cascade.detach().cpu().numpy()
    gold_cascade = gold_cascade.detach().cpu().numpy()
    scores = compute_metrics(pred_cascade, gold_cascade, k_list)
    return scores

def get_scores2(opt, pred_cascade:torch.Tensor, gold_cascade:torch.Tensor, k_list=[10,50,100]):
    gold_cascade = gold_cascade.contiguous().view(-1)           # (#samples,)
    user_num,user_size = pred_cascade.size(0), int(pred_cascade.size(1)/opt.num_heads)
    pred_cascade = pred_cascade.view(user_num,opt.num_heads,user_size).max(1)[0]
    pred_cascade = pred_cascade.detach().cpu().numpy()
    gold_cascade = gold_cascade.detach().cpu().numpy()
    # scores, _ = portfolio(pred_cascade, gold_cascade, k_list)
    scores = compute_metrics(pred_cascade, gold_cascade, k_list)
    return scores

def train(epoch_i, data, graph, model, optimizer, loss_func, writer, log_desc='train_'):
    model.train()

    loss, correct, total = 0., 0., 0.
    for _, batch in enumerate(data['batch']):
    # for _, batch in enumerate(tqdm(data['batch'])):

        cas_users, cas_tss, cas_intervals, cas_classids = batch
        if args.cuda:
            cas_users = cas_users.to(args.gpu)
            cas_tss = cas_tss.to(args.gpu)
            cas_intervals = cas_intervals.to(args.gpu)
        gold_cascade = cas_users[:, 1:]
        
        optimizer.zero_grad()
        if args.model == 'densegat':
            pred_cascade = model(cas_users, cas_intervals, graph)
        elif args.model == 'heteredgegat':
            pred_cascade = model(data['user_side_emb'], cas_users, cas_intervals, cas_classids, data['hedge_graphs'], data['diffusion_graph'], cas_tss,)
        elif args.model == 'diffusiongat':
            pred_cascade = model(cas_users, cas_tss, data['diffusion_graph'])
        elif args.model == 'tan':
            pred_cascade, _ = model((cas_users, cas_intervals, None, None))
        elif args.model == 'dhgpntm':
            pred_cascade = model(cas_users, cas_tss, cas_intervals, None, data['diffusion_graph'])
        elif args.model == 'forest':
            pred_cascade, _ = model(cas_users)
        elif args.model == 'ndm':
            pred_cascade = model(cas_users)
            pred_cascade = pred_cascade[0]
        
        if args.model == 'tan':
            loss_batch, _, n_correct, _ = get_performance2(data['opt'], loss_func, pred_cascade, gold_cascade)
        else:
            loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
        loss_batch.backward()
        optimizer.step()
        optimizer.update_learning_rate()

        n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
        loss += loss_batch.item()
        correct += n_correct.item()
        total += n_words
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1)
    writer.add_scalar(log_desc+'acc',  n_correct/total, epoch_i+1)

    return loss/total, n_correct/total

def evaluate(epoch_i, data, graph, model, optimizer, loss_func, writer, k_list=[10,50,100], log_desc='valid_'):
    model.eval()

    loss, correct, total = 0., 0., 0.
    scores = {'MRR': 0,}
    for k in k_list:
        scores[f'hits@{k}'] = 0
        scores[f'map@{k}'] = 0
    for _, batch in enumerate((data['batch'])):
    # for _, batch in enumerate(tqdm(data['batch'])):

        cas_users, cas_tss, cas_intervals, cas_classids = batch
        if args.cuda:
            cas_users = cas_users.to(args.gpu)
            cas_tss = cas_tss.to(args.gpu)
            cas_intervals = cas_intervals.to(args.gpu)
        gold_cascade = cas_users[:, 1:]
        
        optimizer.zero_grad()
        if args.model == 'densegat':
            pred_cascade = model(cas_users, cas_intervals, graph)
        elif args.model == 'heteredgegat':
            pred_cascade = model(data['user_side_emb'], cas_users, cas_intervals, cas_classids, data['hedge_graphs'], data['diffusion_graph'], cas_tss,)
        elif args.model == 'diffusiongat':
            pred_cascade = model(cas_users, cas_tss, data['diffusion_graph'])
        elif args.model == 'tan':
            pred_cascade, _ = model((cas_users, cas_intervals, None, None))
        elif args.model == 'dhgpntm':
            pred_cascade = model(cas_users, cas_tss, cas_intervals, None, data['diffusion_graph'])
        elif args.model == 'forest':
            pred_cascade, _ = model(cas_users)
        elif args.model == 'ndm':
            pred_cascade = model(cas_users)
            pred_cascade = pred_cascade[0]
        
        if args.model == 'tan':
            loss_batch, _, n_correct, _ = get_performance2(data['opt'], loss_func, pred_cascade, gold_cascade)
        else:
            loss_batch, n_correct = get_performance(loss_func, pred_cascade, gold_cascade)
                
        n_words = (gold_cascade.ne(PAD)*(gold_cascade.ne(EOS))).data.sum().float()
        loss += loss_batch.item()
        correct += n_correct.item()
        total += n_words

        if args.model == 'tan':
            scores_batch = get_scores2(data['opt'], pred_cascade, gold_cascade, k_list)
        else:
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
    # logger.info(f"MRR={scores['MRR']}, hits@10={scores['hits@10']}, map@10={scores['map@10']}, hits@50={scores['hits@50']}, map@100={scores['map@100']}, hits@100={scores['hits@100']}, map@100={scores['map@100']},")
    
    writer.add_scalar(log_desc+'loss', loss/total, epoch_i+1); writer.add_scalar(log_desc+'acc', correct/total, epoch_i+1); writer.add_scalar(log_desc+'mrr', scores['MRR'], epoch_i+1)
    writer.add_scalar(log_desc+'hits@10',  scores['hits@10'],  epoch_i+1);  writer.add_scalar(log_desc+'map@10',  scores['map@10'],  epoch_i+1)
    writer.add_scalar(log_desc+'hits@50',  scores['hits@50'],  epoch_i+1);  writer.add_scalar(log_desc+'map@50',  scores['map@50'],  epoch_i+1)
    writer.add_scalar(log_desc+'hits@100', scores['hits@100'], epoch_i+1);  writer.add_scalar(log_desc+'map@100', scores['map@100'], epoch_i+1)
    # writer.add_scalar(log_desc+'auc', auc, epoch_i+1);                    writer.add_scalar(log_desc+'f1', f1, epoch_i+1)
    # writer.add_scalar(log_desc+'prec', prec, epoch_i+1);                  writer.add_scalar(log_desc+'rec', rec, epoch_i+1)
    
    return scores

def main():
    # torch.set_num_threads(4)

    dataset_dirpath = f"{DATA_ROOTPATH}/{args.dataset}"
    # user_ids = read_user_ids(f"{dataset_dirpath}/train_withcontent.data", f"{dataset_dirpath}/valid_withcontent.data", f"{dataset_dirpath}/test_withcontent.data")
    # edges = get_static_subnetwork(user_ids)
    # _, edges = reindex_edges(user_ids, edges)
    user_edges = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/edges.data"))
    user_edges = list(zip(*user_edges))
    edges_t = torch.LongTensor(user_edges) # (2,#num_edges)
    weight_t = torch.FloatTensor([1]*edges_t.size(1))
    graph = Data(edge_index=edges_t, edge_weight=weight_t)
    if args.cuda:
        graph = graph.to(args.gpu)

    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    train_data = DataConstruct(dataset_dirpath=dataset_dirpath, batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, n_component=args.n_component, data_type=0, load_dict=True)
    valid_data = DataConstruct(dataset_dirpath=dataset_dirpath, batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, n_component=args.n_component, data_type=1, load_dict=True)
    test_data  = DataConstruct(dataset_dirpath=dataset_dirpath, batch_size=args.batch_size, seed=args.seed, tmax=args.tmax, num_interval=args.n_interval, n_component=args.n_component, data_type=2, load_dict=True)

    train_d = {'batch': train_data}; valid_d = {'batch': valid_data}; test_d = {'batch': test_data}

    vertex_feat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/vertex_feature_user{train_data.user_size}.npy"))
    three_sort_feat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/three_sort_feature_user{train_data.user_size}.npy"))
    deepwalk_feat = load_w2v_feature(os.path.join(DATA_ROOTPATH, f"{args.dataset}/feature/deepwalk_emb_user{train_data.user_size}.data"), max_idx=train_data.user_size-1)
    user_side_emb = torch.cat([torch.FloatTensor(vertex_feat),torch.FloatTensor(deepwalk_feat),torch.FloatTensor(three_sort_feat),],dim=1)
    if args.cuda:
        user_side_emb = user_side_emb.to(args.gpu)
    
    new_d = {'user_side_emb': user_side_emb}
    train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

    # TODO: decide on n_feat
    if args.model == 'densegat':
        model = BasicGATNetwork(n_feat=16, n_units=n_units, n_heads=n_heads, num_interval=args.n_interval, shape_ret=(n_units[-1],train_data.user_size), 
            attn_dropout=args.attn_dropout, dropout=args.dropout)
    
    elif args.model == 'heteredgegat':
        # classid2simmat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/llm/classid2simmat_windowsize{args.window_size}.pkl"))
        classid2simmat = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/topic_diffusion_graph_usefull.data"))
        if args.cuda:
            classid2simmat = {classid:simmat.to(args.gpu) for classid, simmat in classid2simmat.items()}
        
        diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_graph.data"))
        if args.cuda:
            diffusion_graph = diffusion_graph[sorted(diffusion_graph.keys())[-1]].to(args.gpu)
        
        n_simmat = max(classid2simmat.keys())+1
        hedge_graphs = [classid2simmat[classid] if classid in classid2simmat else graph for classid in range(n_simmat)] + [graph]

        user_topic_preference = None
        if args.use_topic_preference:
            user_topic_preference = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/llm/user_topic_pref_cnt.pkl"))
            # -1 use mean vector
            user_topic_preference = torch.cat((user_topic_preference, torch.mean(user_topic_preference, dim=1).view(-1,1)), dim=1)
            if args.cuda:
                user_topic_preference = user_topic_preference.to(args.gpu)
        
        new_d = {'hedge_graphs':hedge_graphs, 'user_topic_preference':user_topic_preference, "diffusion_graph": diffusion_graph}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

        model = HeterEdgeGATNetwork(n_feat=32, n_units=n_units, n_heads=n_heads, n_adj=n_simmat, n_comp=args.n_component, num_interval=args.n_interval, shape_ret=(-1,train_data.user_size), 
            attn_dropout=args.attn_dropout, dropout=args.dropout, use_gat=args.use_gat, use_topic_pref=args.use_topic_preference)
    
    elif args.model == 'diffusiongat':
        # diffusion_graph = LoadDynamicHeteGraph(os.path.join(DATA_ROOTPATH, args.dataset))
        diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_graph.data"))
        if args.cuda:
            diffusion_graph = {key:value.to(args.gpu) for key, value in diffusion_graph.items()}
        
        new_d = {'diffusion_graph':diffusion_graph}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

        model = DiffusionGATNetwork(n_feat=64, n_adj=None, n_units=None, n_heads=None, num_interval=len(diffusion_graph), shape_ret=(64,train_data.user_size), 
            attn_dropout=args.attn_dropout, dropout=args.dropout)

    elif args.model == 'tan':
        opt = Option()
        opt.user_size = train_data.user_size
        new_d = {'opt':opt}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)
        model = TAN(opt)
    
    elif args.model == 'dhgpntm':
        # diffusion_graph = LoadDynamicHeteGraph(os.path.join(DATA_ROOTPATH, args.dataset))
        diffusion_graph = load_pickle(os.path.join(DATA_ROOTPATH, f"{args.dataset}/diffusion_graph.data"))
        if args.cuda:
            diffusion_graph = {key:value.to(args.gpu) for key, value in diffusion_graph.items()}
        
        new_d = {'diffusion_graph':diffusion_graph}
        train_d.update(new_d); valid_d.update(new_d); test_d.update(new_d)

        # model = DyHGCN_H(train_data.user_size, 64, 8)
        model = DyHGCN_H(train_data.user_size, 64, args.n_interval)
    
    elif args.model == 'forest':
        model = RNNModel('GRUCell', train_data.user_size, 64, 64)

    elif args.model == 'ndm':
        model = Decoder(train_data.user_size, d_k=64, d_v=64, d_model=64, d_word_vec=64, d_inner_hid=64, n_head=8, kernel_size=3, dropout=0.1) 
    
    # loss_func = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    loss_func = get_cascade_criterion(train_data.user_size)

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 
        args.d_model, args.n_warmup_steps,)
    patience = EarlyStopping(patience=args.patience)
    
    if args.cuda:
        model = model.to(args.gpu)
        loss_func = loss_func.to(args.gpu)
    
    tensorboard_log_dir = '%s/tensorboard-series/tensorboard_%s_epochs%d' % (os.path.dirname(os.path.abspath(__file__)), args.tensorboard_log, args.epochs)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    shutil.rmtree(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

    t_total = time.time()
    logger.info("training...")
    for epoch_i in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train(epoch_i, train_d, graph, model, optimizer, loss_func, writer)
        logger.info('   - (Training)    loss: {loss:8.5f}, accuracy: {accu:3.6f} %, elapse: {elapse:3.3f} min, gpu memory usage: {mem:3.3f} MiB'.format(
            loss=train_loss, accu=100*train_acc, elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
        
        early_stop = patience.step(train_loss.detach().cpu().numpy(), train_acc.detach().cpu().numpy())
        if early_stop:
            start = time.time()
            scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer)
            logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
                scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            save_model(epoch_i, args, model, optimizer)
            break
        
        if (epoch_i + 1) % args.check_point == 0:
            logger.info("epoch %d, checkpoint!", epoch_i)
            valid_loss, valid_acc = train(epoch_i, valid_d, graph, model, optimizer, loss_func, writer)
            logger.info('   - (Validating)    loss: {loss:8.5f}, accuracy: {accu:3.6f} %, gpu memory usage: {mem:3.3f} MiB'.format(
                loss=valid_loss, accu=100*valid_acc, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            
            start = time.time()
            scores = evaluate(epoch_i, test_d, graph, model, optimizer, loss_func, writer)
            logger.info('   - (Testing)    scores: {scores}, elapse: {elapse:3.3f} min, gpu memory usage={mem:3.3f} MiB'.format(
                scores=" ".join([f"{key}:{value:3.6f}" for key,value in scores.items()]),
                elapse=(time.time()-start)/60, mem=check_gpu_memory_usage(int(args.gpu[-1]))))
            save_model(epoch_i, args, model, optimizer)
            
    logger.info("Total Elapse: {elapse:3.3f} min".format(elapse=(time.time()-t_total)/60))

if __name__ == '__main__':
    main()
