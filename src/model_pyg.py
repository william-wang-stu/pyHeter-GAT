from lib.log import logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, GATv2Conv
from utils.Constants import PAD
from src.model import AdditiveAttention
from src.sota.DHGPNTM.TransformerBlock import TransformerBlock
from src.sota.DHGPNTM.DyHGCN import DynamicGraphNN
from typing import Dict

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.to(seq.device)
    
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.to(seq.device)
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.to(seq.device)
    masked_seq = ans_tmp.scatter_(2,masked_seq.long(),float('-inf'))
    return masked_seq

class GATNetwork(nn.Module):
    def __init__(self, n_feat, n_units, n_heads, attn_dropout, dropout,
    ):
        super(GATNetwork, self).__init__()

        self.dropout = dropout
        self.layer_stack, self.batchnorm_stack = self._build_layer_stack(extend_units=[n_feat]+n_units, n_heads=n_heads, attn_dropout=attn_dropout)
        # self.init_weights()

    def init_weights(self):
        for layer in self.layer_stack:
            init.xavier_normal_(layer.weight)
        for batchnorm in self.batchnorm_stack:
            init.xavier_normal_(batchnorm.weight)
    
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout):
        layer_stack = nn.ModuleList()
        batchnorm_stack = nn.ModuleList()
        for layer_i, (n_unit, n_head, f_out, fin_head) in enumerate(zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1])):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            is_last_layer = layer_i == len(extend_units[:-1])-1
            layer_stack.append(
                GATv2Conv(heads=n_head, in_channels=f_in, out_channels=f_out, concat=is_last_layer, dropout=attn_dropout, add_self_loops=True, edge_dim=1),
            )
            if not is_last_layer:
                # batchnorm_stack.append(nn.LayerNorm(f_out))
                batchnorm_stack.append(nn.BatchNorm1d(f_out))
        return layer_stack, batchnorm_stack
    
    def forward(self, graph, emb):
        graph_edge_index = graph.edge_index
        graph_weight = graph.edge_weight

        for layer_i, (gat_layer, batchnorm_layer) in enumerate(zip(self.layer_stack, self.batchnorm_stack)):
            emb = gat_layer(emb, graph_edge_index, graph_weight)
            # emb = gat_layer(x=emb, edge_index=graph_edge_index)
            if layer_i < len(self.layer_stack)-1:
                emb = batchnorm_layer(emb)
                emb = F.elu(emb)
                emb = F.dropout(emb, self.dropout, training=self.training)
        
        return emb.cpu()

class TimeAttention_New(nn.Module):
    def __init__(self, ninterval, nfeat, dropout=0.1):
        super(TimeAttention_New, self).__init__()
        self.time_embedding = nn.Embedding(ninterval, nfeat)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cas_intervals, cas_embs, mask=None, episilon=1e-6):
        temperature = cas_embs.size(-1) ** 0.5 + episilon # d**0.5+eps
        cas_interval_embs = self.time_embedding(cas_intervals)

        affine = torch.einsum("bqd,bkd->bqk", cas_embs, cas_interval_embs)
        score = affine / temperature

        pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, mask.size(1))
        mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool()
        if pad_mask.is_cuda:
            mask = mask.to(pad_mask.device)
        mask_ = mask + pad_mask
        score = score.masked_fill(mask_, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)
        return alpha.bmm(cas_embs)

class BasicGATNetwork(nn.Module):
    def __init__(self, n_feat, n_units, n_heads, num_interval, shape_ret,
        attn_dropout, dropout, instance_normalization=False,
    ):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(BasicGATNetwork, self).__init__()

        self.dropout = dropout
        self.inst_norm = instance_normalization
        
        self.user_size = shape_ret[1]
        self.user_emb = nn.Embedding(self.user_size, n_feat, padding_idx=PAD)
        self.gat_network = GATNetwork(n_feat, n_units, n_heads, attn_dropout, dropout)
        self.time_attention = TimeAttention_New(num_interval=num_interval, in_features1=n_units[-1])
        self.fc_network = nn.Linear(shape_ret[0], shape_ret[1])
        self.init_weights()
    
    def init_weights(self):
        init.xavier_normal_(self.fc_network.weight)
    
    def forward(self, cas_uids, cas_intervals, graph):
        cas_uids = cas_uids[:,:-1]
        cas_intervals = cas_intervals[:,:-1]

        user_emb = self.user_emb(torch.tensor([i for i in range(self.user_size)]).to(cas_uids.device))
        graph_emb = self.gat_network(graph, user_emb)
        seq_embs = F.embedding(cas_uids, graph_emb) # (bs, max_len, D)

        mask = (cas_uids == PAD)
        seq_embs = self.time_attention(cas_intervals, seq_embs, mask)
        seq_embs = F.dropout(seq_embs, self.dropout)

        output = self.fc_network(seq_embs)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output

class DiffusionGATNetwork(nn.Module):
    def __init__(self, n_feat, n_adj, n_units, n_heads, num_interval, shape_ret,
        attn_dropout, dropout,
    ):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(DiffusionGATNetwork, self).__init__()

        # self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.user_size = shape_ret[1]
        # self.user_emb = nn.Embedding(self.user_size, n_feat, padding_idx=PAD)
        self.gnn_diffusion_layer = DynamicGraphNN(self.user_size, shape_ret[0])
        self.ninp = shape_ret[0]
        self.pos_emb_dim = 8
        self.pos_emb = nn.Embedding(1000, self.pos_emb_dim)
        # self.gat_network = GATNetwork(n_feat, n_units, n_heads, attn_dropout, dropout)
        # self.heter_gat_network = nn.ModuleList([
        #     # GATNetwork(n_feat, n_units, n_heads, attn_dropout, dropout)
        #     GPN(num_layers=2, num_mlp_layers=2, input_dim=n_feat, hidden_dim=n_units[0], output_dim=n_units[-1])
        #     for _ in range(n_adj)])
        self.time_attention = TimeAttention_New(ninterval=num_interval, nfeat=shape_ret[0]+self.pos_emb_dim)
        self.decoder_attention = TransformerBlock(input_size=shape_ret[0]+self.pos_emb_dim, n_heads=8)
        self.fc_network = nn.Linear(shape_ret[0]+self.pos_emb_dim, shape_ret[1])
        self.init_weights()
    
    def init_weights(self):
        # init.xavier_normal_(self.user_emb.weight)
        init.xavier_normal_(self.pos_emb.weight)
        init.xavier_normal_(self.fc_network.weight)
    
    def forward(self, cas_uids, cas_tss, diffusion_graph):
        cas_uids = cas_uids[:,:-1]
        cas_tss = cas_tss[:,:-1]

        dynamic_node_emb_dict = self.gnn_diffusion_layer(diffusion_graph) #input, input_timestamp, diffusion_graph) 
        
        batch_size, max_len = cas_uids.size()
        dyemb = torch.zeros(batch_size, max_len, self.ninp).to(cas_uids.device)
        step_len = 1
        
        latest_timestamp = sorted(dynamic_node_emb_dict.keys())[-1]
        for t in range(0, max_len, step_len):
            try:
                la_timestamp = torch.max(cas_tss[:, t:t+step_len]).item()
                if la_timestamp < 1:
                    break 
                latest_timestamp = la_timestamp 
            except Exception:
                # print (input_timestamp[:, t:t+step_len])
                pass 

            his_timestamp = sorted(dynamic_node_emb_dict.keys())[-1]
            for x in sorted(dynamic_node_emb_dict.keys()):
                if x <= latest_timestamp:
                    his_timestamp = x
                    continue
                else:
                    break 

            graph_dynamic_embeddings = dynamic_node_emb_dict[his_timestamp]
            dyemb[:, t:t+step_len, :] = F.embedding(cas_uids[:, t:t+step_len], graph_dynamic_embeddings.to(cas_uids.device))

        # dyemb = F.dropout(dyemb, self.dropout)
        dyemb = self.dropout(dyemb)

        dyemb_timestamp = torch.zeros(batch_size, max_len).long()
        dynamic_node_emb_dict_time = sorted(dynamic_node_emb_dict.keys())
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val] = i
        latest_timestamp = dynamic_node_emb_dict_time[-1]
        for t in range(0, max_len, step_len):
            try:
                la_timestamp = torch.max(cas_tss[:, t:t + step_len]).item()
                if la_timestamp < 1:
                    break
                latest_timestamp = la_timestamp
            except Exception:
                pass

            res_index = len(dynamic_node_emb_dict_time_dict) - 1
            for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                if val <= latest_timestamp:
                    res_index = i
                    continue
                else:
                    break
            dyemb_timestamp[:, t:t + step_len] = res_index

        mask = (cas_uids == PAD)
        batch_t = torch.arange(cas_uids.size(1)).expand(cas_uids.size()).to(cas_uids.device)
        # pos_embs = F.dropout(self.pos_emb(batch_t), self.dropout)
        pos_embs = self.dropout(self.pos_emb(batch_t))

        seq_embs = self.time_attention(dyemb_timestamp.to(cas_uids.device), torch.cat([dyemb, pos_embs], dim=-1), mask)
        # seq_embs = F.dropout(seq_embs, self.dropout)
        # seq_embs = self.dropout(seq_embs)

        seq_embs = self.decoder_attention(seq_embs, seq_embs, seq_embs, mask)
        seq_embs = self.dropout(seq_embs)

        output = self.fc_network(seq_embs) # (bs, max_len, |V|)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output

class HeterEdgeGATNetwork(nn.Module):
    def __init__(self, n_feat, n_adj, n_comp, n_units, n_heads, num_interval, shape_ret,
        attn_dropout, dropout, instance_normalization=False, use_topic_pref=False,
    ):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(HeterEdgeGATNetwork, self).__init__()

        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.user_size = shape_ret[1]
        self.user_emb = nn.Embedding(self.user_size, n_feat, padding_idx=PAD)
        self.pos_emb_dim = 8
        self.pos_emb  = nn.Embedding(1000, self.pos_emb_dim)
        self.heter_gat_network = nn.ModuleList([
            GATNetwork(n_feat, n_units, n_heads, attn_dropout, dropout)
            for _ in range(n_adj+1)])
        # self.gpn_network = nn.ModuleList([
        #     GPN(num_layers=2, num_mlp_layers=2, input_dim=n_feat, hidden_dim=n_units[0], output_dim=n_units[-1])
        #     for _ in range(n_adj)
        # ])
        num_feat = 16
        self.use_topic_pref = use_topic_pref
        if not self.use_topic_pref:
            self.additive_attention = AdditiveAttention(d=n_feat, d1=n_units[-1], d2=n_units[-1])
            self.fc_topic_net = nn.Linear(n_units[-1]*(n_comp+1), num_feat)
        self.time_attention = TimeAttention_New(ninterval=num_interval, nfeat=num_feat+self.pos_emb_dim)
        self.decoder_attention = TransformerBlock(input_size=num_feat+self.pos_emb_dim, n_heads=8)
        self.fc_network = nn.Linear(num_feat+self.pos_emb_dim, shape_ret[1])
        self.init_weights()
    
    def init_weights(self):
        if not self.use_topic_pref:
            init.xavier_normal_(self.fc_topic_net.weight)
        init.xavier_normal_(self.fc_network.weight)
    
    def match_graphemb_with_timestamp(self, heter_user_embs:Dict[int, torch.Tensor], cas_uids:torch.Tensor, cas_timstamps:torch.Tensor, cas_classids:torch.Tensor,):
        bs, ml = cas_timstamps.size()
        n_comp = cas_classids.size(1)
        n_user, emb_size = list(heter_user_embs.values())[0].size()
        step_len = 1
        
        dyemb = torch.zeros(bs, ml, n_comp, emb_size)
        dyemb_timestamp = torch.zeros(bs, ml).long()
        dynamic_node_emb_dict_time = sorted(heter_user_embs.keys(), key=lambda x:x[0])
        dynamic_node_emb_dict_time_dict = dict()
        for i, val in enumerate(dynamic_node_emb_dict_time):
            dynamic_node_emb_dict_time_dict[val[0]] = i

        latest_timestamp = dynamic_node_emb_dict_time[-1][0]
        for t in range(0, ml, step_len):
            try:
                la_timestamp = torch.max(cas_timstamps[:, t:t+step_len]).item()
                if la_timestamp < 1:
                    break
                latest_timestamp = la_timestamp
            except Exception:
                pass

            his_timestamp = dynamic_node_emb_dict_time[-1][0]
            his_interval  = len(dynamic_node_emb_dict_time_dict)-1
            for i, (x,_) in enumerate(dynamic_node_emb_dict_time):
                if x <= latest_timestamp:
                    his_timestamp = x
                    his_interval  = i
                else:
                    break
            
            # TODO: slowwwww...
            for batch_i in range(bs):
                topic_aware_graph_embs = torch.cat([heter_user_embs[(his_timestamp,classid)].unsqueeze(1) if (his_timestamp,classid) in heter_user_embs else heter_user_embs[(-1,-1)].unsqueeze(1) for classid in cas_classids[batch_i]], dim=1)
                dyemb[batch_i, t:t+step_len, :] = F.embedding(cas_uids[batch_i,t:t+step_len].cpu(), topic_aware_graph_embs.reshape(n_user,-1)).reshape(1,step_len,n_comp,emb_size)
            dyemb_timestamp[:, t:t+step_len] = his_interval
        
        dyemb = dyemb.reshape(-1,n_comp,emb_size)
        dyemb = F.dropout(dyemb, self.dropout)
        if cas_uids.is_cuda:
            dyemb = dyemb.to(cas_uids.device)
            dyemb_timestamp = dyemb_timestamp.to(cas_uids.device)
        return dyemb, dyemb_timestamp
    
    def forward2(self, cas_uids, cas_intervals, cas_classids:torch.Tensor, hedge_graphs, cas_tss=None, diffusion_graph_keys=None, user_topic_preference:torch.Tensor=None):
        assert len(hedge_graphs) == len(self.heter_gat_network)

        cas_uids = cas_uids[:,:-1]
        cas_intervals = cas_intervals[:,:-1]
        if cas_tss is not None:
            cas_tss = cas_tss[:,:-1]

        user_emb = self.user_emb(torch.tensor([i for i in range(self.user_size)]).to(cas_uids.device))
        heter_user_embs = []
        for heter_i, gat_network in enumerate(self.heter_gat_network):
            graph_emb = gat_network(hedge_graphs[heter_i], user_emb)
            # graph_emb = gpn_layer(user_emb, hedge_graphs[heter_i].edge_index, hedge_graphs[heter_i].edge_weight)
            heter_user_embs.append(graph_emb.unsqueeze(1))  # (Nu,    1, D')
        topic_aware_embs = torch.cat(heter_user_embs, dim=1) # (Nu, |Rs|, D')

        bs, ml = cas_uids.size()[:2]
        selected_aware_seq_embs = torch.zeros(bs, ml, cas_classids.size(1), topic_aware_embs.size(-1)) # (bs, max_len, n_comp, D')
        for batch_i in range(bs):
            aware_seq_embs = topic_aware_embs[:,cas_classids[batch_i],:].reshape(topic_aware_embs.size(0),-1)
            selected_aware_seq_embs[batch_i] = F.embedding(cas_uids[batch_i].cpu(), aware_seq_embs).reshape(1,ml,cas_classids.size(1),-1)
        n_comp, d = selected_aware_seq_embs.size()[2:]
        selected_aware_seq_embs = selected_aware_seq_embs.view(-1,n_comp,d) # (bs*max_len, n_comp, D')
        if cas_uids.is_cuda:
            selected_aware_seq_embs = selected_aware_seq_embs.to(cas_uids.device)
        
        if user_topic_preference is None:
            user_seq_embs = F.embedding(cas_uids, user_emb)
            user_seq_embs = user_seq_embs.view(-1,user_seq_embs.size(-1)) # (bs*max_len, D)
            assert user_seq_embs.size(0) == selected_aware_seq_embs.size(0)

            fusion_seq_embs = self.additive_attention(user_seq_embs, selected_aware_seq_embs) # (bs*max_len, 1, D')
            seq_embs = torch.cat((selected_aware_seq_embs, fusion_seq_embs),dim=1).reshape(bs,ml,-1) # (bs, max_len, (n_comp+1)*D')
            seq_embs = self.fc_topic_net(seq_embs) # (bs,ml,D')
        else:
            topic_prefer_seqs = F.embedding(cas_uids, user_topic_preference)
            selected_topic_prefer_seqs = torch.zeros(bs, ml, cas_classids.size(1))
            if topic_prefer_seqs.is_cuda:
                selected_topic_prefer_seqs = selected_topic_prefer_seqs.to(topic_prefer_seqs.device)
            for batch_i in range(aware_seq_embs.size(0)):
                selected_topic_prefer_seqs[batch_i] = topic_prefer_seqs[batch_i, :, cas_classids[batch_i],]
            selected_topic_prefer_seqs = selected_topic_prefer_seqs.view(bs*ml,-1,1)
            assert selected_aware_seq_embs.size(0) == selected_topic_prefer_seqs.size(0)
            
            fusion_seq_embs = (selected_aware_seq_embs*selected_topic_prefer_seqs).sum(dim=1).view(bs*ml, -1).unsqueeze(1) # (bs*max_len, 1, D')
            seq_embs = torch.cat((selected_aware_seq_embs, fusion_seq_embs),dim=1).reshape(bs,ml,-1) # (bs, max_len, (n_comp+1)*D')
        seq_embs = F.dropout(seq_embs, self.dropout)

        if diffusion_graph_keys is not None:
            batch_size, max_len = cas_uids.size()
            step_len = 1
            dyemb_timestamp = torch.zeros(batch_size, max_len).long()
            dynamic_node_emb_dict_time = sorted(diffusion_graph_keys)
            dynamic_node_emb_dict_time_dict = dict()
            for i, val in enumerate(dynamic_node_emb_dict_time):
                dynamic_node_emb_dict_time_dict[val] = i
            latest_timestamp = dynamic_node_emb_dict_time[-1]
            for t in range(0, max_len, step_len):
                try:
                    la_timestamp = torch.max(cas_tss[:, t:t + step_len]).item()
                    if la_timestamp < 1:
                        break
                    latest_timestamp = la_timestamp
                except Exception:
                    pass

                res_index = len(dynamic_node_emb_dict_time_dict) - 1
                for i, val in enumerate(dynamic_node_emb_dict_time_dict.keys()):
                    if val <= latest_timestamp:
                        res_index = i
                        continue
                    else:
                        break
                dyemb_timestamp[:, t:t + step_len] = res_index

        batch_t = torch.arange(cas_uids.size(1)).expand(cas_uids.size()).to(cas_uids.device)
        pos_embs = F.dropout(self.pos_emb(batch_t), self.dropout)

        mask = (cas_uids == PAD)
        if diffusion_graph_keys is not None:
            seq_embs = self.time_attention(dyemb_timestamp.to(cas_uids.device), torch.cat([seq_embs, pos_embs], dim=-1), mask)
        else:
            seq_embs = self.time_attention(cas_intervals, torch.cat([seq_embs, pos_embs], dim=-1), mask)
        seq_embs = F.dropout(seq_embs, self.dropout)

        seq_embs = self.decoder_attention(seq_embs, seq_embs, seq_embs, mask)
        output = self.fc_network(seq_embs) # (bs, max_len, |V|)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output

    def forward(self, cas_uids, cas_intervals, cas_classids:torch.Tensor, graph, hedge_graphs_dict:dict, cas_tss=None, ):
        cas_uids = cas_uids[:,:-1]
        cas_intervals = cas_intervals[:,:-1]
        if cas_tss is not None:
            cas_tss = cas_tss[:,:-1]

        user_emb = self.user_emb(torch.tensor([i for i in range(self.user_size)]).to(cas_uids.device))
        heter_user_embs = {}
        for graph_i, ts_key in enumerate(hedge_graphs_dict.keys()):
            gat_network = self.heter_gat_network[graph_i]
            hedge_graph = hedge_graphs_dict[ts_key]
            heter_user_embs[ts_key] = gat_network(hedge_graph, user_emb)
        heter_user_embs[(-1,-1)] = self.heter_gat_network[-1](graph, user_emb)
        
        topic_aware_seq_embs, topic_aware_tss = self.match_graphemb_with_timestamp(heter_user_embs, cas_uids, cas_tss, cas_classids)
        
        user_seq_embs = F.embedding(cas_uids, user_emb)
        user_seq_embs = user_seq_embs.view(-1,user_seq_embs.size(-1)) # (bs*max_len, D)
        assert user_seq_embs.size(0) == topic_aware_seq_embs.size(0)

        fusion_seq_embs = self.additive_attention(user_seq_embs, topic_aware_seq_embs) # (bs*max_len, 1, D')
        seq_embs = torch.cat((topic_aware_seq_embs, fusion_seq_embs),dim=1).reshape(bs,ml,-1) # (bs, max_len, (n_comp+1)*D')
        seq_embs = self.fc_topic_net(seq_embs) # (bs,ml,D')
        seq_embs = F.dropout(seq_embs, self.dropout)

        batch_t = torch.arange(cas_uids.size(1)).expand(cas_uids.size()).to(cas_uids.device)
        pos_embs = F.dropout(self.pos_emb(batch_t), self.dropout)

        mask = (cas_uids == PAD)
        seq_embs = self.time_attention(topic_aware_tss, torch.cat([seq_embs, pos_embs], dim=-1), mask)
        seq_embs = F.dropout(seq_embs, self.dropout)

        seq_embs = self.decoder_attention(seq_embs, seq_embs, seq_embs, mask)
        output = self.fc_network(seq_embs) # (bs, max_len, |V|)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output
