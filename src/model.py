import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from lib.log import logger

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
        self, n_user, pretrained_emb,
        nb_node_kinds=2, nb_classes=2, n_units=[25,64], n_heads=[3], d2=64,
        attn_dropout=0.5, dropout=0.1, 
        fine_tune=False, instance_normalization=False
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
        
        self.d  = n_units[0]
        self.d1 = n_units[1]
        self.d2 = n_units[1]
        self.n_user = n_user

        self.layer_stack = nn.ModuleList()
        for _ in range(nb_node_kinds):
            layer_stack = nn.ModuleList()
            for i in range(self.n_layer):
                f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
                layer_stack.append(
                    SpGATLayer(n_head=n_heads[i], f_in=f_in, f_out=n_units[i+1], attn_dropout=attn_dropout)
                )
            self.layer_stack.append(layer_stack)
        self.additive_attention = AdditiveAttention(d=self.d, d1=self.d1, d2=self.d2)
        self.fc_layer = nn.Linear(in_features=self.d1*(nb_node_kinds+1), out_features=nb_classes)
    
    def forward(self, h: torch.Tensor, vertices, hadj: torch.Tensor): # h:(n, fin), vertices:(n,), hadj:(|Rs|,n,n)
        emb = self.embedding(vertices[:self.n_user])
        if self.inst_norm:
            emb = self.norm1(emb.transpose(1, 2)).transpose(1, 2)
        emb = torch.cat((emb, torch.empty(vertices.shape[0]-emb.size(0), emb.size(1)).fill_(0).to(emb.device)), dim=0) # (n_user, f_emb) -> (n, f_emb)
        h = torch.cat((h, emb), dim=1)
        
        n = h.shape[0]
        heter_embs = []
        for heter_idx, layer_stack in enumerate(self.layer_stack):
            x = h
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(x, hadj[heter_idx]) # output: (n, n_head, f_out)
                if i + 1 == self.n_layer:
                    x = x.mean(dim=1) # (n, n_head, f_out) -> (n, f_out)
                else:
                    x = F.elu(x.reshape(n, -1)) # (n, n_head, f_out) -> (n, n_head*f_out)
                    x = F.dropout(x, self.dropout, training=self.training)
            heter_embs.append(x[:self.n_user].unsqueeze(-2)) # (Nu, 1, f_out)
        type_aware_emb = torch.cat(heter_embs, dim=-2) # (Nu, |Rs|, D')
        type_fusion_emb = self.additive_attention(h[:self.n_user], type_aware_emb) # (Nu, 1, D')
        ret = self.fc_layer(
            torch.cat((type_aware_emb, type_fusion_emb),dim=1).reshape(self.n_user,-1), # (Nu, |Rs|+1, D') -> (Nu, (|Rs|+1)*D')
        ) #  (Nu, nb_classes)
        return F.log_softmax(ret, dim=-1)
