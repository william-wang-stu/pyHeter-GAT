from lib.log import logger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, GATv2Conv
from typing import List
from utils.Constants import PAD
from src.model import AdditiveAttention

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
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
        
        return emb

class TimeAttention_New(nn.Module):
    def __init__(self, num_interval, in_features1):
        super(TimeAttention_New, self).__init__()
        self.time_embedding = nn.Embedding(num_interval, in_features1)
        init.xavier_normal_(self.time_embedding.weight)
        self.dropout = nn.Dropout(0.3)

    def forward(self, T_idx, Dy_U_embed, mask=None, episilon=1e-6):
        '''
            T_idx: (bsz, user_len)
            Dy_U_embed: (bsz, user_len, d) # uid 从动态embedding lookup 之后的节点向量
            output: (bsz, user_len, d)
        '''
        temperature = Dy_U_embed.size(-1) ** 0.5 + episilon
        T_embed = self.time_embedding(T_idx) # (bsz, user_len, d)

        affine = torch.einsum("bqd,bkd->bqk", Dy_U_embed, T_embed) # (bsz, user_len, time_len)
        score = affine / temperature

        pad_mask = mask.unsqueeze(dim=-1).expand(-1, -1, mask.size(1))
        mask = torch.triu(torch.ones(pad_mask.size()), diagonal=1).bool()  # 上三角
        if pad_mask.is_cuda:
            mask = mask.cuda()
        mask_ = mask + pad_mask
        score = score.masked_fill(mask_, -2 ** 32 + 1)

        alpha = F.softmax(score, dim=1)  # (bsz, user_len, time_len)

        att = alpha.bmm(Dy_U_embed)
        return att

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
        # if self.inst_norm:
        #     for i, dynamic_nfeat in enumerate(dynamic_nfeats):
        #         norm = nn.InstanceNorm1d(dynamic_nfeat, momentum=0.0, affine=True)
        #         setattr(self, f"norm-{i}", norm)
        self.user_size = shape_ret[1]
        self.user_emb = nn.Embedding(self.user_size, n_feat, padding_idx=PAD)
        self.gat_network = GATNetwork(n_feat, n_units, n_heads, attn_dropout, dropout)
        self.time_attention = TimeAttention_New(num_interval=num_interval, in_features1=n_units[0])
        self.fc_network = nn.Linear(shape_ret[0], shape_ret[1])
        self.init_weights()
    
    def init_weights(self):
        init.xavier_normal_(self.fc_network.weight)
    
    def forward(self, cas_uids, cas_intervals, graph):
    # def forward(self, cas_uids, cas_intervals, graph, static_emb:torch.Tensor, dynamic_embs:List[torch.Tensor]):
        # norm_embs = []
        # for i, dynamic_emb in enumerate(dynamic_embs):
        #     norm_emb = dynamic_emb
        #     if self.inst_norm:
        #         norm = getattr(self, f"norm-{i}")
        #         norm_emb = norm(norm_emb.transpose(0,1)).transpose(0,1)
        #     norm_embs.append(norm_emb)
        # emb = torch.cat(norm_embs, dim=1)
        # emb = torch.cat((emb,static_emb),dim=1)
        cas_uids = cas_uids[:,:-1]
        cas_intervals = cas_intervals[:,:-1]

        user_emb = self.user_emb(torch.tensor([i for i in range(self.user_size)]))
        graph_emb = self.gat_network(graph, user_emb)
        seq_embs = F.embedding(cas_uids, graph_emb)

        mask = (cas_uids == PAD)
        seq_embs = self.time_attention(cas_intervals, seq_embs, mask)
        seq_embs = F.dropout(seq_embs, self.dropout)

        output = self.fc_network(seq_embs)
        mask = get_previous_user_mask(cas_uids, self.user_size)
        output = output + mask
        return output
