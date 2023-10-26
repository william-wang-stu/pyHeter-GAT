from utils.log import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import List

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=False):
        super(BatchMultiHeadGraphAttention, self).__init__()

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

    def forward(self, h:torch.Tensor, adj:torch.Tensor):
        bs, n = h.size()[:2] # bs x n x f_in
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

    def forward(self, feature:torch.Tensor, type_aware_emb:torch.Tensor):
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

class BatchDenseGAT(nn.Module):
    def __init__(
        self, global_embs, n_feat, n_units, n_heads, 
        attn_dropout, dropout, instance_normalization=False, norm_mask=None, fine_tune=False,
    ) -> None:
        """
        Arguments:
            global_embs: List[emb]
            n_units: contains hidden unit dimension of each layer
            n_heads: contains attention head number of each layer
            shape_ret: (n_user,nb_classes=2), contains shape of required return tensor
            norm_mask: List[Bool], shape=len(global_embs),1
        """
        super().__init__()

        self.dropout = dropout
        self.inst_norm = instance_normalization

        self.norm_mask = norm_mask
        for i, global_emb in enumerate(global_embs):
            emb = nn.Embedding(global_emb.size(0), global_emb.size(1))
            emb.weight = nn.Parameter(global_emb)
            emb.weight.requires_grad = fine_tune
            setattr(self, f"emb-{i}", emb)

            if self.inst_norm:
                norm = nn.InstanceNorm1d(global_emb.size(1), momentum=0.0, affine=True)
                setattr(self, f"norm-{i}", norm)

        self.layer_stack = self._build_layer_stack(extend_units=[n_feat]+n_units, n_heads=n_heads, attn_dropout=attn_dropout)
        # self.fc_layer = nn.Linear(in_features=n_units[-1], out_features=shape_ret[1])
    
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                BatchMultiHeadGraphAttention(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
        return layer_stack

    def forward(self, adj:torch.Tensor, vertices:torch.Tensor, local_emb:torch.Tensor):
        # adj: bs*n*n, emb: bs*n*n_feat
        global_embs = []
        for i, mask in enumerate(self.norm_mask):
            emb = getattr(self, f"emb-{i}")
            global_emb = emb(vertices)
            if self.inst_norm and mask:
                norm = getattr(self, f"norm-{i}")
                global_emb = norm(global_emb.transpose(1,2)).transpose(1,2)
            global_embs.append(global_emb)
        emb = torch.cat(global_embs, dim=2)
        emb = torch.cat((emb, local_emb), dim=2)

        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            emb = gat_layer(emb, adj) # bs*n_head*n*f_out
            if i+1 == len(self.layer_stack):
                emb = emb.mean(dim=1) # bs*n*f_out
            else:
                emb = F.elu(emb.transpose(1,2).contiguous().view(bs, n, -1)) # bs*n*(n_head*f_out)
                emb = F.dropout(emb, self.dropout, training=self.training)
        # emb = self.fc_layer(emb) # bs*n*shape_ret[1]
        return F.log_softmax(emb, dim=-1)

class HeterEdgeDenseGAT(nn.Module):
    def __init__(self, global_embs, n_feat, n_adj, n_units, n_heads, 
        attn_dropout, dropout, instance_normalization=False, norm_mask=None, fine_tune=False,
    ):
        """
        shape_ret: (n_units[-1], #user)
        """
        super(HeterEdgeDenseGAT, self).__init__()

        self.dropout = dropout
        self.inst_norm = instance_normalization
        # self.user_size = shape_ret[1]

        self.norm_mask = norm_mask
        for i, global_emb in enumerate(global_embs):
            emb = nn.Embedding(global_emb.size(0), global_emb.size(1))
            emb.weight = nn.Parameter(global_emb)
            emb.weight.requires_grad = fine_tune
            setattr(self, f"emb-{i}", emb)

            if self.inst_norm:
                norm = nn.InstanceNorm1d(global_emb.size(1), momentum=0.0, affine=True)
                setattr(self, f"norm-{i}", norm)

        # self.pos_emb_dim = 8
        # self.pos_emb  = nn.Embedding(1000, self.pos_emb_dim)

        self.layer_stack = nn.ModuleList([
            self._build_layer_stack(extend_units=[n_feat]+n_units, n_heads=n_heads, attn_dropout=attn_dropout)
            for _ in range(n_adj)])

        self.additive_attention = BatchAdditiveAttention(d=n_feat, d1=n_units[-1], d2=n_units[-1])
        # self.time_attention = TimeAttention_New(ninterval=num_interval, nfeat=last_dim*(n_dim+1)+self.pos_emb_dim)
            
    def _build_layer_stack(self, extend_units, n_heads, attn_dropout):
        layer_stack = nn.ModuleList()
        for n_unit, n_head, f_out, fin_head in zip(extend_units[:-1], n_heads, extend_units[1:], [None]+n_heads[:-1]):
            f_in = n_unit*fin_head if fin_head is not None else n_unit
            layer_stack.append(
                BatchMultiHeadGraphAttention(n_head=n_head, f_in=f_in, f_out=f_out, attn_dropout=attn_dropout),
            )
        return layer_stack
    
    def forward(self, hedge_graphs:torch.Tensor, vertices:torch.Tensor, local_emb:torch.Tensor):
        # hadjs: K*bs*n*n, emb: bs*n*n_feat
        global_embs = []
        for i, mask in enumerate(self.norm_mask):
            emb = getattr(self, f"emb-{i}")
            global_emb = emb(vertices)
            if self.inst_norm and mask:
                norm = getattr(self, f"norm-{i}")
                global_emb = norm(global_emb.transpose(1,2)).transpose(1,2)
            global_embs.append(global_emb)
        emb = torch.cat(global_embs, dim=2)
        emb = torch.cat((emb, local_emb), dim=2)

        assert len(hedge_graphs) >= 1
        bs, _, n = hedge_graphs.size()[:3]
        h_embs = []
        for h_i, layer_stack in enumerate(self.layer_stack):
            adj = hedge_graphs[:,h_i]
            h_emb = emb.clone()
            for i, gat_layer in enumerate(layer_stack):
                h_emb = gat_layer(h_emb, adj) # bs*n_head*n*f_out
                if i+1 == len(layer_stack):
                    h_emb = h_emb.mean(dim=1) # bs*n*f_out
                else:
                    h_emb = F.elu(h_emb.transpose(1,2).contiguous().view(bs, n, -1)) # bs*n*(n_head*f_out)
                    h_emb = F.dropout(h_emb, self.dropout, training=self.training)
            h_embs.append(h_emb.unsqueeze(2))
        seq_embs = torch.cat(h_embs, dim=2)
        seq_embs = F.dropout(seq_embs, self.dropout)
        
        fusion_seq_embs = self.additive_attention(emb, seq_embs) # (bs, max_len, 1, D')
        # seq_embs = torch.mean(
        #     torch.cat((seq_embs, fusion_seq_embs), dim=2), dim=2)
        seq_embs = fusion_seq_embs.squeeze(2)
        return F.log_softmax(seq_embs, dim=-1)
