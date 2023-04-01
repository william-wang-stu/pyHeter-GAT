# NOTE: https://stackoverflow.com/a/56806766
# import sys
# import os
# sys.path.append(os.path.dirname(os.getcwd()))
from lib.log import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from dgl.nn.pytorch import GATConv, GATv2Conv
from typing import List

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()

        self.n_head = n_head
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        n = h.size(0) # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w) #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src) # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_dst) # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1) # n_head x n x n

        attn = self.leaky_relu(attn)
        attn.data.masked_fill_(~adj, float("-inf"))
        attn = self.softmax(attn) # n_head x n x n
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime).transpose(0,1) # n x n_head x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output

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
            self.bias = nn.Parameter(torch.Tensor(f_out))
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

class AdditiveAttention(nn.Module):
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
            feature: initial feature of all main nodes, shape is (N, D)
            type_aware_emb: ~, shape is (N, |Rs|, D')
        Return:
            type_fusion_emb: ~, shape is (N, 1, D')
        """
        _, nb_node_kind, _ = type_aware_emb.shape
        feature = feature.unsqueeze(-2).repeat(1, nb_node_kind, 1) # (N, |Rs|, D)

        q = self.tanh(self.w1(feature)+self.w2(type_aware_emb)) # (N, |Rs|, D'')
        q = self.m(q) # (N, |Rs|, 1)
        beta = self.softmax(q.squeeze(-1)) # (N, |Rs|)
        type_fusion_emb = torch.bmm(
            beta.unsqueeze(-2), type_aware_emb #(N,1,|Rs|) * (N,|Rs|,D') -> (N,1,D')
        )
        return type_fusion_emb

class HeterSparseGAT(nn.Module):
    """
    思路: 多邻接矩阵+稀疏注意力头
    特点: 先把不同类型节点映射到统一特征空间维度中
    """
    def __init__(
        self, n_feats, n_unified, n_units, n_heads, shape_ret,
        attn_dropout, dropout, instance_normalization=False, sparse=True, skip_fc=False,
    ) -> None:
        """
        Arguments:
            n_feats: [f1,f2,...,fn], contains initial dimensions of all node types
            n_units: contains hidden unit dimension of each layer
            n_heads: contains attention head number of each layer
            shape_ret: (n_user,nb_classes=2), contains shape of required return tensor
        """
        super().__init__()

        self.shape_ret = shape_ret
        self.dropout = dropout
        self.inst_norm = instance_normalization
        self.skip_fc = skip_fc
        d1 = n_units[-1]
        for i, n_feat in enumerate(n_feats):
            weight_mat = nn.Parameter(torch.Tensor(n_feat, n_unified))
            nn.init.xavier_uniform_(weight_mat) # set initial values
            setattr(self, f"weight-mat-{i}", weight_mat)
            if self.inst_norm:
                setattr(self, f"norm-{i}", nn.InstanceNorm1d(n_feat, momentum=0.0, affine=True))
        
        self.layer_stack = nn.ModuleList([
                self._build_layer_stack(extend_units=[n_unified]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse)
            for _ in n_feats])
        self.additive_attention = AdditiveAttention(d=n_unified, d1=d1, d2=n_units[-1])
        self.fc_layer = nn.Linear(in_features=d1*(len(n_feats)+1), out_features=shape_ret[1])

    def _build_layer_stack(self, extend_units, n_heads, attn_dropout, sparse):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                SpGATLayer(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout) if sparse else
                    MultiHeadGraphAttention(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
            # f_in = extend_units[idx] * n_heads[idx-1] if idx else extend_units[idx]
            # layer_stack.append(
            #     SpGATLayer(n_head=n_heads[idx], f_in=f_in, f_out=extend_units[idx+1], attn_dropout=attn_dropout) if sparse else
            #         MultiHeadGraphAttention(n_head=n_heads[idx], f_in=f_in, f_out=extend_units[idx+1], attn_dropout=attn_dropout),
            # )
        return layer_stack
    
    def forward(self, hadjs: torch.Tensor, hembs: List[torch.Tensor]): # hadj:(|Rs|,n,n), hembs:(|Rs|,n,f_in), f_in1!=f_in2
        trans_embs:List[torch.Tensor] = []
        for i, hemb in enumerate(hembs):
            weight_mat = getattr(self, f"weight-mat-{i}")
            trans_emb = torch.matmul(hemb, weight_mat) # (n,f_in) -> (n,f_in_unified)
            if self.inst_norm:
                norm = getattr(self, f"norm-{i}")
                trans_emb = norm(trans_emb.transpose(0,1)).transpose(0,1)
            trans_embs.append(trans_emb)

        heter_embs = []
        for heter_idx, layer_stack in enumerate(self.layer_stack):
            x, hadj = trans_embs[heter_idx], hadjs[heter_idx] # x: (n,f_in_unified)
            n = x.shape[0]
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(x, hadj) # (n,n_head,f_out)
                if i+1 == len(layer_stack):
                    x = x.mean(dim=1) # (n,n_head,f_out) -> (n,f_out)
                else:
                    x = F.elu(x.reshape(n, -1)) # (n,n_head,f_out) -> (n,n_head*f_out)
                    x = F.dropout(x, self.dropout, training=self.training)
            heter_embs.append(x[:self.shape_ret[0]].unsqueeze(-2)) # (Nu, 1, f_out)
        type_aware_emb = torch.cat(heter_embs, dim=-2) # (Nu, |Rs|, D')
        type_fusion_emb = self.additive_attention(trans_emb[0], type_aware_emb) # (Nu, 1, D')
        ret = torch.cat((type_aware_emb, type_fusion_emb),dim=1).reshape(self.shape_ret[0],-1) # (Nu, |Rs|+1, D') -> (Nu, (|Rs|+1)*D')
        if self.skip_fc:
            return ret
        ret = self.fc_layer(ret) #  (Nu, nb_classes)
        return F.log_softmax(ret, dim=-1)

class HyperGAT(nn.Module):
    """
    思路: 用户侧特征+用户侧邻接矩阵做GAT卷积, 推文侧特征+推文侧邻接矩阵做GAT卷积, 最后合并concat两侧表征
    """
    def __init__(
        self, n_feats, n_units, n_heads, shape_ret,
        attn_dropout, dropout, instance_normalization=False, sparse=True,
        wo_user_centralized_net=False, wo_tweet_centralized_net=False,
    ) -> None:
        super().__init__()

        self.shape_ret = shape_ret
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            for i, n_feat in enumerate(n_feats):
                setattr(self, f"norm-{i}", nn.InstanceNorm1d(n_feat, momentum=0.0, affine=True))

        # NOTE: User-Centralized & Tweet-Centralized GAT-Network
        self.layer_stack = nn.ModuleList()
        self.layer_mask = [1-int(wo_user_centralized_net), 1-int(wo_tweet_centralized_net)]
        for i, mask in enumerate(self.layer_mask):
            if mask == 1: self.layer_stack.append(
                    self._build_layer_stack(extend_units=[n_feats[i]]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse)
                )
        self.fc_layer = nn.Linear(in_features=n_units[-1]*(2-int(wo_user_centralized_net)-int(wo_tweet_centralized_net)), out_features=shape_ret[1])
    
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout, sparse):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                SpGATLayer(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout) if sparse else
                    MultiHeadGraphAttention(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
        return layer_stack
    
    def forward(self, hadjs:List[torch.Tensor], hembs:List[torch.Tensor]):
        if self.inst_norm:
            norm_embs:List[torch.Tensor] = []
            for i, (emb,mask) in enumerate(zip(hembs,self.layer_mask)):
                if mask == 0: continue
                norm = getattr(self, f"norm-{i}")
                norm_emb = norm(emb.transpose(0,1)).transpose(0,1)
                norm_embs.append(norm_emb)
            hembs = norm_embs

        heter_embs = []
        for stack_idx, layer_stack in enumerate(self.layer_stack):
            emb, adj = hembs[stack_idx], hadjs[stack_idx]
            n = emb.shape[0]
            for layer_idx, gat_layer in enumerate(layer_stack):
                emb = gat_layer(emb, adj) # (n, n_head, f_out)
                if layer_idx+1 == len(layer_stack):
                    emb = emb.mean(dim=1) # (n, n_head, f_out) -> (n, f_out)
                else:
                    emb = F.elu(emb.reshape(n, -1)) # (n, n_head, f_out) -> (n, n_head*f_out)
                    emb = F.dropout(emb, self.dropout, training=self.training)
            heter_embs.append(emb[:self.shape_ret[0]]) # (n, f_out) -> (n_user, f_out)
        ret = torch.cat(heter_embs, dim=-1) # (n_user, f_out)*2 -> (n_user, f_out*2)
        ret = self.fc_layer(ret) # (n_user, nb_classes)
        return F.log_softmax(ret, dim=-1)

class HyperGATWithHeterSparseGAT(nn.Module):
    """
    思路: 在HyperGAT的基础上, 用HeterSparseGAT替代Tweet-Centralized-Network
    """
    def __init__(
        self, n_feats, n_unified, n_units, n_heads, shape_ret,
        attn_dropout, dropout, instance_normalization=False, sparse=True,
        wo_user_centralized_net=False,
    ) -> None:
        super().__init__()

        self.shape_ret = shape_ret
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            for i, n_feat in enumerate(n_feats):
                setattr(self, f"norm-{i}", nn.InstanceNorm1d(n_feat, momentum=0.0, affine=True))

        # NOTE: User-Centralized & Tweet-Centralized GAT-Network
        self.layer_stack = nn.ModuleList()
        self.layer_mask = [1-int(wo_user_centralized_net), 1]
        for i, mask in enumerate(self.layer_mask):
            if i == 0 and mask == 1: self.layer_stack.append(
                    self._build_layer_stack(extend_units=[n_feats[i]]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse)
                )
            elif i == 1 and mask == 1: self.layer_stack.append(
                    HeterSparseGAT(n_feats=n_feats, n_unified=n_unified, n_units=n_units, n_heads=n_heads, shape_ret=shape_ret,
                        attn_dropout=attn_dropout, dropout=dropout, instance_normalization=instance_normalization, sparse=sparse, skip_fc=True)
                )
        self.fc_layer = nn.Linear(in_features=n_units[-1]*(1-int(wo_user_centralized_net)+len(n_feats)+1), out_features=shape_ret[1])
    
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout, sparse):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                SpGATLayer(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout) if sparse else
                    MultiHeadGraphAttention(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
        return layer_stack
    
    def forward(self, hadjs:List[torch.Tensor], hembs:List[torch.Tensor]):
        """
        hadjs: [Nu*Nu, [(Nu+Nt)*(Nu+Nt),(Nu+Nt)*(Nu+Nt)]],
        hembs: [Nu*fu, (Nu+Nt)*(fu+ft)],
        """
        if self.inst_norm:
            norm_embs:List[torch.Tensor] = []
            for i, (emb,mask) in enumerate(zip(hembs,self.layer_mask)):
                if mask == 0: continue
                norm = getattr(self, f"norm-{i}")
                norm_emb = norm(emb.transpose(0,1)).transpose(0,1)
                norm_embs.append(norm_emb)
            hembs = norm_embs

        heter_embs = []
        for stack_idx, layer_stack in enumerate(self.layer_stack):
            emb, adj = hembs[stack_idx], hadjs[stack_idx]
            if stack_idx == 0 and self.layer_mask[stack_idx] == 1:
                # User-Centralized Network (Using Multi-Stack GAT)
                n = emb.shape[0]
                for layer_idx, gat_layer in enumerate(layer_stack):
                    emb = gat_layer(emb, adj) # (n, n_head, f_out)
                    if layer_idx+1 == len(layer_stack):
                        emb = emb.mean(dim=1) # (n, n_head, f_out) -> (n, f_out)
                    else:
                        emb = F.elu(emb.reshape(n, -1)) # (n, n_head, f_out) -> (n, n_head*f_out)
                        emb = F.dropout(emb, self.dropout, training=self.training)
                assert emb.shape[0] == self.shape_ret[0]
                heter_embs.append(emb[:self.shape_ret[0]]) # (n, f_out) -> (n_user, f_out)
            else:
                # Tweet-Centralized Network (Using HeterSparseGAT)
                heter_embs.append(layer_stack(adj, emb))
        ret = torch.cat(heter_embs, dim=-1) # (n_user, f_out)*2 -> (n_user, f_out*2)
        ret = self.fc_layer(ret) # (n_user, nb_classes)
        return F.log_softmax(ret, dim=-1)
