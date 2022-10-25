import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import copy
from lib.log import logger

class SpGATLayer(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=False):
        super(SpGATLayer, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out

        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.attn_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.attn_trg = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=attn_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_trg)

    def explicit_broadcast(self, this, other):
        """from https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb"""
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        
        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    
    def special_spmm(self, other, trg_indices, size):
        ret = torch.zeros(size=size, dtype=other.dtype, device=other.device)
        trg_indices_broadcast = self.explicit_broadcast(trg_indices, other)
        ret.scatter_add_(dim=0, index=trg_indices_broadcast, src=other)
        return ret
    
    def calc_neigh_attn(self, e, trg_indices, nb_nodes):
        e = e - e.max()
        exp_e = e.exp()
        neigh_attn = exp_e / (self.special_spmm(other=exp_e, trg_indices=trg_indices, size=(nb_nodes, self.n_head)).index_select(0, trg_indices)+1e-16)
        return neigh_attn.unsqueeze(-1) # (E,k)->(E,k,1)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # NOTE: h: (n, fin), adj: (n, n)
        n = h.shape[0]
        h_prime  = torch.matmul(h.unsqueeze(0), self.w) # (k,n,f_out)
        h_prime  = self.dropout(h_prime)
        attn_src = torch.matmul(h_prime, self.attn_src).sum(dim=-1).permute(1,0) # (n,k)
        attn_trg = torch.matmul(h_prime, self.attn_trg).sum(dim=-1).permute(1,0) # (n,k)

        (src_indices, trg_indices) = adj._indices()
        attn_src_lifted, attn_trg_lifted, h_prime_lifted = attn_src[src_indices], attn_trg[trg_indices], h_prime.permute(1,0,2)[src_indices]
        e = self.leaky_relu(attn_src_lifted+attn_trg_lifted) # (E,k)

        neigh_attn = self.calc_neigh_attn(e=e, trg_indices=trg_indices, nb_nodes=n) # (E,k,1)
        neigh_attn = self.dropout(neigh_attn)

        h_weighted = h_prime_lifted*neigh_attn # (E,k,f_out)
        out = self.special_spmm(h_weighted, trg_indices=trg_indices, size=(n, self.n_head, self.f_out)) # (n,k,f_out)

        if self.bias is not None:
            return out + self.bias
        else:
            return out

class BatchSparseMultiHeadGraphAttention(nn.Module):
    def __init__(self, nb_heads, nb_in_feats, nb_out_feats, nb_loop_nodes, attn_dropout, bias=False):
        super(BatchSparseMultiHeadGraphAttention, self).__init__()
        self.nb_heads = nb_heads
        self.nb_in_feats = nb_in_feats
        self.nb_out_feats = nb_out_feats
        self.nb_loop_nodes = nb_loop_nodes

        self.linear = nn.Linear(in_features=nb_in_feats, out_features=nb_heads*nb_out_feats, bias=False)
        self.attn_src = nn.Parameter(torch.Tensor(1, nb_heads, nb_out_feats))
        self.attn_trg = nn.Parameter(torch.Tensor(1, nb_heads, nb_out_feats))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=attn_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(nb_out_feats))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        
        self.init_params()
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_trg)

    def add_self_connections_to_edge_index(self, edge_index: torch.Tensor):
        device = edge_index.device
        self_loop = torch.arange(self.nb_loop_nodes, dtype=int).repeat(2,1).to(device)
        edge_index_ = torch.cat((edge_index, self_loop), dim=1)
        return edge_index_
    
    def explicit_broadcast(self, this, other):
        """from https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb"""
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        
        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    
    def sum_neigh_e(self, exp_e, trg_indices, nb_nodes):
        trg_indices_broadcast = self.explicit_broadcast(trg_indices, exp_e)
        trg_norm = torch.zeros(size=(nb_nodes, self.nb_heads), dtype=exp_e.dtype, device=exp_e.device)
        trg_norm.scatter_add_(dim=0, index=trg_indices_broadcast, src=exp_e)
        return trg_norm.index_select(0, trg_indices)

    def calc_neigh_attn(self, e, trg_indices, nb_nodes):
        e = e - e.max()
        exp_e = e.exp()
        neigh_attn = exp_e / (self.sum_neigh_e(exp_e, trg_indices, nb_nodes)+1e-16)
        return neigh_attn.unsqueeze(-1) # (E,k)->(E,k,1)

    def aggregate_neigh(self, h_weighted, trg_indices, h_dtype, h_device, nb_nodes):
        out = torch.zeros(size=(nb_nodes, self.nb_heads, self.nb_out_feats), dtype=h_dtype, device=h_device)
        trg_indices_broadcast = self.explicit_broadcast(trg_indices, h_weighted)
        out.scatter_add_(dim=0, index=trg_indices_broadcast, src=h_weighted)
        return out.permute(1,0,2) # (n,k,nb_out_feats) -> (k,n,nb_out_feats)

    def forward(self, h: torch.Tensor, adjs: torch.Tensor):
        # NOTE: h: (bs, n, fin), adjs: (bs, n, n)
        _, n = h.shape[:2]
        # logger.info(f"{h.shape}, {adjs.shape}")

        h_prime  = self.linear(h.unsqueeze(1)).view(-1, n, self.nb_heads, self.nb_out_feats) # (bs,1,n,nb_in_feats)*(nb_in_feats,k*nb_out_feats) -> (bs,n,k,nb_out_feats)
        attn_src = (h_prime*self.attn_src).sum(dim=-1) # ( (bs,n,k,nb_out_feats)*(1,k,nb_out_feats) ).sum(-1) -> (bs, n, k)
        attn_trg = (h_prime*self.attn_trg).sum(dim=-1)

        out_batch = []
        for i_batch, adj in enumerate(adjs):
            edge_index = adj._indices()
            edge_index_ = edge_index
            # edge_index_ = self.add_self_connections_to_edge_index(edge_index)

            # attn_src_lifted = attn_src[i_batch].index_select(0, edge_index_[0]) # (E,k)
            # attn_trg_lifted = attn_trg[i_batch].index_select(0, edge_index_[1])
            # h_prime_lifted  = h_prime[i_batch].index_select(0, edge_index_[0])  # (E,k,nb_out_feats)
            attn_src_lifted = attn_src[i_batch][edge_index_[0]] # (E,k)
            attn_trg_lifted = attn_trg[i_batch][edge_index_[1]]
            h_prime_lifted  = h_prime[i_batch][edge_index_[0]]  # (E,k,nb_out_feats)

            e = self.leaky_relu(attn_src_lifted+attn_trg_lifted)
            neigh_attn = self.calc_neigh_attn(e, edge_index_[1], n)
            neigh_attn = self.dropout(neigh_attn)

            h_weighted = h_prime_lifted*neigh_attn
            out = self.aggregate_neigh(h_weighted, trg_indices=edge_index_[1], h_dtype=h.dtype, h_device=h.device, nb_nodes=n)
            out_batch.append(out)
        out_batch = torch.stack(out_batch) # (bs, K, N, nb_out_feats)
        if self.bias is not None:
            return out_batch + self.bias
        else:
            return out_batch

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=False):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2] # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w) # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src) # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst) # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2) # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = 1 - adj.unsqueeze(1) # bs x 1 x n x n
        attn.data.masked_fill_(mask.bool(), float("-inf"))
        attn = self.softmax(attn) # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime) # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias, attn_dst
        else:
            return output

class BatchAdditiveAttention(nn.Module):
    def __init__(self, d, d1, d2) -> None:
        super().__init__()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.w1 = nn.Linear(in_features=d, out_features=d2, bias=False)
        self.w2 = nn.Linear(in_features=d1, out_features=d2, bias=False)
        self.m  = nn.Linear(in_features=d2, out_features=1, bias=False)
    
    def forward(self, feature: torch.Tensor, type_aware_emb: torch.Tensor):
        """
        Args:
            feature: initial feature of all main nodes, shape is (bs, N, D)
            type_aware_emb: ~, shape is (bs, N, |Rs|, D')
        Return:
            type_fusion_emb: ~, shape is (bs, N, 1, D')
        """
        bs, n, nb_node_kind, d1 = type_aware_emb.shape
        feature = feature.unsqueeze(-2).repeat(1, 1, nb_node_kind, 1) # (bs, N, |Rs|, D)

        q = self.tanh(self.w1(feature)+self.w2(type_aware_emb)) # (bs, N, |Rs|, D'')
        q = self.m(q) # (bs, N, |Rs|, 1)
        beta = self.softmax(q.squeeze(-1)) # (bs, N, |Rs|)
        type_fusion_emb = torch.bmm(
            beta.unsqueeze(-2).view(-1,1,nb_node_kind), type_aware_emb.view(-1,nb_node_kind,d1) #(bs*N,1,|Rs|) * (bs*N,|Rs|,D') -> (bs*N,1,D')
        ).view(bs, n, 1, d1) # (bs, N, 1, D')
        return type_fusion_emb

class HeterGraphAttentionNetwork(nn.Module):
    def __init__(
        self, n_user, nb_node_kinds=2, nb_loop_nodes=[50,1050],
        nb_classes=2, n_units=[25,64], n_heads=[3],
        attn_dropout=0.5, dropout=0.1, 
        d2=64, gpu_device_ids=[] 
    ) -> None:
        """
        Args:
            gpu_device_ids(List[int], default=[]): 采用Model Parallel方法, 主要使多头注意力可以在不同GPU上运行, 
                模型以数据所属GPU为例, 先在该参数指定的不同GPU上执行单一注意力头, 再在最后将所有注意力头的运行结果复制回数据所属的主GPU上
        """
        super().__init__()

        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.gpu_device_ids = gpu_device_ids

        self.d = n_units[0]
        self.d1 = n_units[1]
        self.d2 = n_units[1]
        self.n_user = n_user

        self.layer_stack = nn.ModuleList()
        for hidx in range(nb_node_kinds):
            layer_stack = nn.ModuleList()
            for i in range(self.n_layer):
                # consider multi head from last layer
                f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
                layer_stack.append(
                    # BatchSparseMultiHeadGraphAttention(nb_heads=n_heads[i], nb_in_feats=f_in, nb_out_feats=n_units[i + 1], 
                    #     nb_loop_nodes=nb_loop_nodes[hidx], attn_dropout=attn_dropout)
                    BatchMultiHeadGraphAttention(n_head=n_heads[i], f_in=f_in, f_out=n_units[i+1], attn_dropout=attn_dropout)
                )
            self.layer_stack.append(layer_stack)
        self.additive_attention = BatchAdditiveAttention(d=self.d, d1=self.d1, d2=self.d2)
        # self.fc_layer = nn.Linear(in_features=self.d1*(nb_node_kinds+1), out_features=nb_classes)
        self.fc_layer = nn.Linear(in_features=self.d1*nb_node_kinds, out_features=nb_classes)
    
    def forward(self, h, hadj):
        # NOTE: h: (bs, N, fin), hadj: (|Rs|, bs, N, N)
        bs, n = h.shape[:2]
        heter_embs = []
        for heter_idx, layer_stack in enumerate(self.layer_stack):
            x = copy.deepcopy(h)
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(x, hadj[heter_idx]) # output: (bs, n_head, n, f_out)
                if i + 1 == self.n_layer:
                    x = x.mean(dim=-3) # (bs, n_head, n, f_out) -> (bs, n, f_out)
                else:
                    x = F.elu(x.reshape(bs, n, -1))
                    x = F.dropout(x, self.dropout, training=self.training)
            heter_embs.append(x[:,:self.n_user].unsqueeze(-2)) # (bs, Nu, 1, f_out)
        type_aware_emb = torch.cat(heter_embs, dim=-2) # (bs, Nu, |Rs|, D')
        # type_fusion_emb = self.additive_attention(h[:,:self.n_user], type_aware_emb) # (bs, Nu, 1, D')
        ret = self.fc_layer(
            # torch.cat((type_fusion_emb, type_aware_emb), dim=-2).reshape(bs, self.n_user,-1) # (bs, Nu, |Rs|+1, D') -> (bs, Nu, (|Rs|+1)*D')
            type_aware_emb.reshape(bs, self.n_user,-1) # (bs, Nu, |Rs|+1, D') -> (bs, Nu, (|Rs|+1)*D')
        ) #  (bs, Nu, nb_classes)
        return F.log_softmax(ret, dim=-1)

class BatchdenseGAT(nn.Module):
    def __init__(
        self, pretrained_emb,
        n_units=[25,64], n_heads=[3],
        attn_dropout=0.5, dropout=0.1, 
        fine_tune=False, instance_normalization=True, use_user_emb=True,
    ) -> None:
        super().__init__()

        self.n_layer = len(n_units) - 1
        self.dropout = dropout

        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm1 = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
        
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)

        self.use_user_emb = use_user_emb
        if self.use_user_emb:
            n_units[0] += 3 # user_feats
            if self.inst_norm:
                self.norm2 = nn.InstanceNorm1d(3, momentum=0.0, affine=True)

        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                    BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                        f_out=n_units[i + 1], attn_dropout=attn_dropout)
                    )
    
    def forward(self, vertices, adj, h, user_emb=None):
        emb = self.embedding(vertices)
        if self.inst_norm:
            emb = self.norm1(emb.transpose(1, 2)).transpose(1, 2)
        h = torch.cat((h, emb), dim=2)
        if self.use_user_emb:
            if self.inst_norm:
                user_emb = self.norm2(user_emb.transpose(1,2)).transpose(1, 2)
            h = torch.cat((h, user_emb), dim=2)

        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            h = gat_layer(h, adj) # bs x n_head x n x f_out
            if i + 1 == self.n_layer:
                h = h.mean(dim=1)
            else:
                h = F.elu(h.transpose(1, 2).contiguous().view(bs, n, -1))
                h = F.dropout(h, self.dropout, training=self.training)
        return F.log_softmax(h, dim=-1)

class HeterdenseGAT(nn.Module):
    def __init__(
        self, n_user, pretrained_emb,
        nb_node_kinds=2, nb_classes=2, n_units=[25,64], n_heads=[3], d2=64,
        attn_dropout=0.5, dropout=0.1, 
        fine_tune=False, instance_normalization=True, use_user_emb=True,
    ) -> None:
        super().__init__()

        self.n_layer = len(n_units) - 2
        self.dropout = dropout

        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm1 = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
        
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += pretrained_emb.size(1)

        self.use_user_emb = use_user_emb
        if self.use_user_emb:
            n_units[0] += 3 # user_feats
            if self.inst_norm:
                self.norm2 = nn.InstanceNorm1d(3, momentum=0.0, affine=True)

        self.d = n_units[0]
        self.d1 = n_units[1]
        self.d2 = n_units[1]
        self.n_user = n_user

        self.layer_stack = nn.ModuleList()
        for _ in range(nb_node_kinds):
            layer_stack = nn.ModuleList()
            for i in range(self.n_layer):
                f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
                layer_stack.append(
                    BatchMultiHeadGraphAttention(n_head=n_heads[i], f_in=f_in, f_out=n_units[i+1], attn_dropout=attn_dropout)
                )
            self.layer_stack.append(layer_stack)
        self.additive_attention = BatchAdditiveAttention(d=self.d, d1=self.d1, d2=self.d2)
        self.fc_layer = nn.Linear(in_features=self.d1*(nb_node_kinds+1), out_features=nb_classes)
    
    def forward(self, vertices, hadj, h, user_emb=None):
        # NOTE: h: (bs, n, fin), vertices: (bs, n), hadj: (|Rs|, bs, n, n)
        emb = self.embedding(vertices[:,:self.n_user])
        if self.inst_norm:
            emb = self.norm1(emb.transpose(1, 2)).transpose(1, 2)
        emb = torch.cat((emb, torch.empty(emb.size(0), vertices.shape[1]-emb.size(1), emb.size(2)).fill_(0).to(emb.device)), dim=1) # (bs, n_user, f_emb) -> (bs, n, f_emb)
        h = torch.cat((h, emb), dim=2)
        if self.use_user_emb:
            if self.inst_norm:
                user_emb = self.norm2(user_emb.transpose(1,2)).transpose(1, 2)
            h = torch.cat((h, user_emb), dim=2)
    
        bs, n = h.shape[:2]
        heter_embs = []
        for heter_idx, layer_stack in enumerate(self.layer_stack):
            x = h
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(x, hadj[heter_idx]) # output: (bs, n_head, n, f_out)
                if i + 1 == self.n_layer:
                    x = x.mean(dim=-3) # (bs, n_head, n, f_out) -> (bs, n, f_out)
                else:
                    x = F.elu(x.reshape(bs, n, -1))
                    x = F.dropout(x, self.dropout, training=self.training)
            heter_embs.append(x[:,:self.n_user].unsqueeze(-2)) # (bs, Nu, 1, f_out)
        type_aware_emb = torch.cat(heter_embs, dim=-2) # (bs, Nu, |Rs|, D')
        type_fusion_emb = self.additive_attention(h[:,:self.n_user], type_aware_emb) # (bs, Nu, 1, D')
        ret = self.fc_layer(
            torch.cat((type_aware_emb, type_fusion_emb),dim=2).reshape(bs, self.n_user,-1), # (bs, Nu, |Rs|+1, D') -> (bs, Nu, (|Rs|+1)*D')
        ) #  (bs, Nu, nb_classes)
        return F.log_softmax(ret, dim=-1)
