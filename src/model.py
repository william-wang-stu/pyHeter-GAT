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

class HetersparseGAT(nn.Module):
    def __init__(
        self, n_user, nb_node_kinds, nb_classes,
        n_units, n_heads, attn_dropout, dropout,
        use_pretrained_emb=False, pretrained_emb=None, fine_tune=False, instance_normalization=False, 
        sparse=True, skip_fc=False,
    ) -> None:
        super().__init__()

        self.n_user = n_user
        self.dropout = dropout
        self.skip_fc = skip_fc
        self.use_pretrained_emb = use_pretrained_emb
        self.inst_norm = instance_normalization
        if self.use_pretrained_emb:
            # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
            self.embedding = nn.Embedding(pretrained_emb.size(0), pretrained_emb.size(1))
            self.embedding.weight = nn.Parameter(pretrained_emb)
            self.embedding.weight.requires_grad = fine_tune
            n_units[0] += pretrained_emb.size(1)
            
            if self.inst_norm:
                self.norm = nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
        
        self.layer_stack = nn.ModuleList([
                self._build_layer_stack(extend_units=n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse) # n_units=[feat.shape[1]]+n_units
            for _ in range(nb_node_kinds)])

        d1 = n_units[1]
        self.additive_attention = AdditiveAttention(d=n_units[0], d1=d1, d2=n_units[1])
        self.fc_layer = nn.Linear(in_features=d1*(nb_node_kinds+1), out_features=nb_classes)

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
    
    def forward(self, hadj: torch.Tensor, h: torch.Tensor, vertices=None): # h:(n, fin), vertices:(n,), hadj:(|Rs|,n,n)
        if self.use_pretrained_emb and vertices is not None:
            emb = self.embedding(vertices[:self.n_user])
            if self.inst_norm:
                emb = self.norm(emb.transpose(0, 1)).transpose(0, 1)
            emb = torch.cat((emb, torch.empty(vertices.shape[0]-emb.size(0), emb.size(1)).fill_(0).to(emb.device)), dim=0) # (n_user, f_emb) -> (n, f_emb)
            h = torch.cat((h, emb), dim=1)
        
        n = h.shape[0]
        heter_embs = []
        for heter_idx, layer_stack in enumerate(self.layer_stack):
            x = h
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(x, hadj[heter_idx]) # output: (n, n_head, f_out)
                if i+1 == len(layer_stack):
                    x = x.mean(dim=1) # (n, n_head, f_out) -> (n, f_out)
                else:
                    x = F.elu(x.reshape(n, -1)) # (n, n_head, f_out) -> (n, n_head*f_out)
                    x = F.dropout(x, self.dropout, training=self.training)
            heter_embs.append(x[:self.n_user].unsqueeze(-2)) # (Nu, 1, f_out)
        type_aware_emb = torch.cat(heter_embs, dim=-2) # (Nu, |Rs|, D')
        type_fusion_emb = self.additive_attention(h[:self.n_user], type_aware_emb) # (Nu, 1, D')
        ret = torch.cat((type_aware_emb, type_fusion_emb),dim=1).reshape(self.n_user,-1) # (Nu, |Rs|+1, D') -> (Nu, (|Rs|+1)*D')
        if self.skip_fc:
            return ret
        ret = self.fc_layer(ret) #  (Nu, nb_classes)
        return F.log_softmax(ret, dim=-1)

class HetergatWOConcatFeat(nn.Module):
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

class HyperGraphAttentionNetwork(nn.Module):
    def __init__(
        self, n_user, f_dims, nb_classes,
        n_units, n_heads, attn_dropout, dropout,
        instance_normalization=False, sparse=True,
        wo_user_centralized_net=False, wo_tweet_centralized_net=False,
    ) -> None:
        super().__init__()

        self.n_user = n_user
        self.dropout = dropout
        self.f_dims = f_dims

        self.inst_norm = instance_normalization
        if self.inst_norm:
            for vec_idx, vecspace_dim in enumerate(self.f_dims):
                setattr(self, f"inst-norm-id{vec_idx}-dim{vecspace_dim}", nn.InstanceNorm1d(vecspace_dim, momentum=0.0, affine=True))

        # NOTE: User-Centralized & Tweet-Centralized GAT-Network
        self.layer_stack = nn.ModuleList()
        if not wo_user_centralized_net:
            self.layer_stack.append(
                self._build_layer_stack(n_layer=len(n_units), extend_units=[f_dims[0]]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse))
        else:
            self.layer_stack.append(None)
        if not wo_tweet_centralized_net:
            self.layer_stack.append(
                self._build_layer_stack(n_layer=len(n_units), extend_units=[f_dims[1]]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse))
        else:
            self.layer_stack.append(None)
        self.fc_layer = nn.Linear(in_features=n_units[-1]*(2-int(wo_user_centralized_net)-int(wo_tweet_centralized_net)), out_features=nb_classes)
    
    def _build_layer_stack(self, n_layer, extend_units, n_heads, attn_dropout, sparse):
        layer_stack = nn.ModuleList()
        for idx in range(n_layer):
            f_in = extend_units[idx] * n_heads[idx-1] if idx else extend_units[idx]
            layer_stack.append(
                SpGATLayer(n_head=n_heads[idx], f_in=f_in, f_out=extend_units[idx+1], attn_dropout=attn_dropout) if sparse else
                    MultiHeadGraphAttention(n_head=n_heads[idx], f_in=f_in, f_out=extend_units[idx+1], attn_dropout=attn_dropout),
            )
        return layer_stack
    
    def forward(self, hadjs, hembs):
        if self.inst_norm:
            norm_embs = []
            for vec_idx, vecspace_dim in enumerate(self.f_dims):
                norm = getattr(self, f"inst-norm-id{vec_idx}-dim{vecspace_dim}")
                norm_emb = norm(hembs[vec_idx].transpose(0,1)).transpose(0,1)
                norm_embs.append(norm_emb)
            hembs = norm_embs

        ret = []
        for stack_idx, layer_stack in enumerate(self.layer_stack):
            if layer_stack is None:
                continue
            emb, adj = hembs[stack_idx], hadjs[stack_idx]
            n = emb.shape[0]
            for layer_idx, gat_layer in enumerate(layer_stack):
                emb = gat_layer(emb, adj) # (n, n_head, f_out)
                if layer_idx+1 == len(layer_stack):
                    emb = emb.mean(dim=1) # (n, n_head, f_out) -> (n, f_out)
                else:
                    emb = F.elu(emb.reshape(n, -1)) # (n, n_head, f_out) -> (n, n_head*f_out)
                    emb = F.dropout(emb, self.dropout, training=self.training)
            ret.append(emb[:self.n_user,:]) # (n, f_out) -> (n_user, f_out)
        ret = torch.cat(ret, dim=-1) # (n_user, f_out)*2 -> (n_user, f_out*2)
        ret = self.fc_layer(ret) # (n_user, nb_classes)
        return F.log_softmax(ret, dim=-1)

class ClusterHeterGATNetwork(nn.Module):
    def __init__(
        self, n_user, f_dims, nb_classes,
        n_units, n_heads, attn_dropout, dropout,
        instance_normalization=False, sparse=True, wo_user_centralized_net=False,
    ) -> None:
        super().__init__()

        self.n_user = n_user
        self.dropout = dropout
        self.f_dims = f_dims

        self.inst_norm = instance_normalization
        if self.inst_norm:
            for vec_idx, vecspace_dim in enumerate(self.f_dims):
                setattr(self, f"inst-norm-id{vec_idx}-dim{vecspace_dim}", nn.InstanceNorm1d(vecspace_dim, momentum=0.0, affine=True))

        self.layer_stack = nn.ModuleList()
        if not wo_user_centralized_net:
            self.layer_stack.append(
                self._build_layer_stack(n_layer=len(n_units), extend_units=[f_dims[0]]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, sparse=sparse))
        self.layer_stack.append(
            HetersparseGAT(n_user=n_user, nb_node_kinds=len(f_dims), nb_classes=2, 
                n_units=[f_dims[0]+f_dims[1]]+n_units, n_heads=n_heads, attn_dropout=attn_dropout, dropout=dropout, 
                instance_normalization=instance_normalization, sparse=sparse, skip_fc=True))
        fc_dim = n_units[-1]*(1+len(f_dims)+1) if not wo_user_centralized_net \
            else n_units[-1]*(len(f_dims)+1)
        self.fc_layer = nn.Linear(in_features=fc_dim, out_features=nb_classes)
    
    def _build_layer_stack(self, n_layer, extend_units, n_heads, attn_dropout, sparse):
        layer_stack = nn.ModuleList()
        for idx in range(n_layer):
            f_in = extend_units[idx] * n_heads[idx-1] if idx else extend_units[idx]
            layer_stack.append(
                SpGATLayer(n_head=n_heads[idx], f_in=f_in, f_out=extend_units[idx+1], attn_dropout=attn_dropout) if sparse else
                    MultiHeadGraphAttention(n_head=n_heads[idx], f_in=f_in, f_out=extend_units[idx+1], attn_dropout=attn_dropout),
            )
        return layer_stack
    
    def forward(self, hadjs, hembs):
        """
        hadjs: [Nu*Nu, [(Nu+Nt)*(Nu+Nt),(Nu+Nt)*(Nu+Nt)]],
        hembs: [Nu*fu, (Nu+Nt)*(fu+ft)],
        """
        if self.inst_norm:
            norm_embs = []
            for vec_idx, vecspace_dim in enumerate(self.f_dims):
                norm = getattr(self, f"inst-norm-id{vec_idx}-dim{vecspace_dim}")
                norm_emb = norm(hembs[vec_idx].transpose(0,1)).transpose(0,1)
                norm_embs.append(norm_emb)
            hembs = norm_embs

        ret = []
        if len(self.layer_stack) == 1:
            layer_stack, adj, emb = self.layer_stack[0], hadjs[1], hembs[1]
            ret.append(layer_stack(adj, emb))
        elif len(self.layer_stack) == 2:
            # User-Centralized Network
            layer_stack, adj, emb = self.layer_stack[0], hadjs[0], hembs[0]
            n = emb.shape[0]
            for layer_idx, gat_layer in enumerate(layer_stack):
                emb = gat_layer(emb, adj) # (n, n_head, f_out)
                if layer_idx+1 == len(layer_stack):
                    emb = emb.mean(dim=1) # (n, n_head, f_out) -> (n, f_out)
                else:
                    emb = F.elu(emb.reshape(n, -1)) # (n, n_head, f_out) -> (n, n_head*f_out)
                    emb = F.dropout(emb, self.dropout, training=self.training)
            ret.append(emb[:self.n_user,:]) # (n, f_out) -> (n_user, f_out)
            
            # Tweet-Centralized Network
            layer_stack, adj, emb = self.layer_stack[1], hadjs[1], hembs[1]
            ret.append(layer_stack(adj, emb).squeeze(1)) # (n_user, 1, f_out) -> (n_user, f_out)
        
        ret = self.fc_layer(torch.cat(ret, dim=-1)) # (n_user, nb_classes)
        return F.log_softmax(ret, dim=-1)
