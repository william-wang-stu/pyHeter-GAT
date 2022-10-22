import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# from log import logger

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(MultiHeadGraphAttention, self).__init__()
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
        n = h.size(0) # h is of size n x f_in
        h_prime = torch.matmul(h.unsqueeze(0), self.w) #  n_head x n x f_out
        attn_src = torch.bmm(h_prime, self.a_src) # n_head x n x 1
        attn_dst = torch.bmm(h_prime, self.a_dst) # n_head x n x 1
        attn = attn_src.expand(-1, -1, n) + attn_dst.expand(-1, -1, n).permute(0, 2, 1) # n_head x n x n

        attn = self.leaky_relu(attn)
        attn.data.masked_fill_(1 - adj, float("-inf"))
        attn = self.softmax(attn) # n_head x n x n
        attn = self.dropout(attn)
        output = torch.bmm(attn, h_prime) # n_head x n x f_out

        if self.bias is not None:
            return output + self.bias
        else:
            return output


# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)

#     # NOTE: STILL cant fix the 89.73GB problem
#     @staticmethod
#     def backward(ctx, grad_output):
#         # grad_output: shape is (N, D')
#         # b: shape is also (N, D'), which points to h_ in forward2
#         # 155195*128
#         # REASON: grad_output.matmul(b.t()) creates a N*N matrix, which comsumes 155195^2*4/1024^3~89GB
#         a, b = ctx.saved_tensors
#         logger.info(b.shape)

#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b


# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)


class SparseMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=False):
        super(SparseMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        
        self.f_in = f_in
        self.f_out = f_out

        # NOTE: forward2()
        # self.w = nn.Parameter(torch.Tensor(size=(n_head, f_in, f_out)))
        # nn.init.xavier_normal_(self.w.data, gain=1.414)
        # self.a = nn.Parameter(torch.Tensor(size=(n_head, 1, 2*f_out)))
        # nn.init.xavier_normal_(self.a.data, gain=1.414)
        # self.special_spmm = SpecialSpmm()

        # self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        # self.a = Parameter(torch.Tensor(n_head, 2*f_out, 1))

        self.lin = nn.Linear(f_in, n_head*f_out)
        self.a_src = nn.Parameter(torch.Tensor(1, n_head, f_out))
        self.a_trg = nn.Parameter(torch.Tensor(1, n_head, f_out))
        
        init.xavier_uniform_(self.lin.weight)
        init.uniform_(self.lin.bias)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_trg)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        # init.xavier_uniform_(self.w)
        # init.xavier_uniform_(self.a)

    # def forward(self, h: torch.Tensor, adj: torch.Tensor):
    #     n = h.size(0)
    #     h_prime = torch.matmul(h.unsqueeze(0), self.w) # (n_head, N, f_out)
    #     from_, to_ = adj._indices()[0], adj._indices()[1] # |from_|, |to_| = E
    #     output_l = []
    #     for head_idx in range(self.n_head):
    #         h_prime_, a_ = h_prime[head_idx], self.a[head_idx] # (N, f_out)
    #         # NOTE:
    #         # logger.info(f"head_idx={head_idx} Allocated: {torch.cuda.memory_reserved(0)/1024**3}")
    #         h_ = torch.cat([h_prime_[from_], h_prime_[to_]], dim=1) # (E, 2*f_out)
    #         # logger.info("Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
    #         attn = torch.mm(h_, a_) # (E, 1)
    #         attn = self.leaky_relu(attn)
    
    #         sparse_attn = torch.sparse_coo_tensor(adj._indices(), attn[:,0], size=(n,n))
    #         # sparse_attn = torch.sparse.FloatTensor(adj._indices(), attn[:,0], torch.Size((n,n)))
    #         attn = torch.sparse.softmax(sparse_attn, dim=1)
            
    #         indices, values = attn.coalesce().indices(), attn.coalesce().values()
    #         values = self.dropout(values)
    #         attn = torch.sparse_coo_tensor(indices, values, size=(n,n), requires_grad=False)
    #         # logger.info("Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #         # NOTE: cause huge GPU Memory Usage during backward()
    #         # output = torch.spmm(attn, h_prime_)
    #         output = torch.sparse.mm(attn, h_prime_)

    #         output_l.append(output.unsqueeze(0))
    #         # # logger.info("Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
    #     ret = torch.cat(output_l, dim=0) # (n_head, N, f_out)
        
    #     if self.bias is not None:
    #         return ret + self.bias
    #     else:
    #         return ret

    def add_self_connections_to_edge_index(self, edge_index: torch.Tensor, N):
        device = edge_index.device
        self_loop = torch.arange(N, dtype=int).repeat(2,1).to(device)
        edge_index_ = torch.cat((edge_index, self_loop), dim=1)
        return edge_index_
    
    def explicit_broadcast(self, this, other):
        """from https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb"""
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            # logger.info("unsqueeze")
            this = this.unsqueeze(-1)
        # Explicitly expand so that shapes are the same
        return this.expand_as(other)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        n = h.shape[0]  # number of nodes
        edge_index = adj._indices()
        # the input data excludes self-connections but [gat] uses them. So we need to add them
        edge_index_ = self.add_self_connections_to_edge_index(edge_index, n)

        h_prime = self.lin(h).view(n, self.n_head, self.f_out)  # (N, F_IN)*(F_IN, K*F_OUT) -> (N, K*F_OUT) -> (N, K, F_OUT)
        h_src = (h_prime[edge_index_[0]] * self.a_src).sum(dim=2)  # ((N, K, F_OUT) * (1, K, F_OUT)).sum(2) -> (N, K)
        h_trg = (h_prime[edge_index_[0]] * self.a_trg).sum(dim=2)

        # compute raw attention coefficients `e_{ij}` in paper [gat, p.3]
        e = h_src + h_trg  # (E, K)
        e = self.leaky_relu(e)

        # apply softmax normalization over all source nodes per target node
        e = e - e.max()   # trick to improve numerical stability before computing exponents (for softmax)
        exp_e = e.exp()  # = unnormalized attention for each pair self.edge_pair[0]->self.edge_pair[1]
        
        trg_normalization = torch.zeros(size=(n, self.n_head), dtype=h.dtype, device=h.device)  # tensor with normalizing constants for each target node
        index = self.explicit_broadcast(edge_index_[1], exp_e)
        trg_normalization.scatter_add(dim=0, index=index, src=exp_e)   # index1:[E, K], exp_e:[E, K] -> [N, K]

        # normalized attention coefficients `alpha_{ij}` in paper [gat, p.3]
        alpha = exp_e / (trg_normalization[edge_index_[1]] + 1e-10)  # (E, K), s.t. for a given target, the sum of the sources = 1.

        # alpha = dropout(alpha)
        src = h_prime[edge_index_[0]]  # (E, K, F_OUT)
        src = src * alpha.unsqueeze(-1)
        out = torch.zeros(size=(n, self.n_head, self.f_out), dtype=h.dtype, device=h.device)
        index = self.explicit_broadcast(edge_index_[1], src)
        out = out.scatter_add(0, index, src)  # (N, K, F_OUT)  # h double prime
        out = out.permute(1,0,2)

        if self.bias is not None:
            return out + self.bias
        else:
            return out

    # NOTE: Analyze GPU Memory Usage
    # def forward(self, h: torch.Tensor, adj: torch.Tensor):
    #     # logger.info("Forward() Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     n = h.shape[0]  # number of nodes
    #     edge_index = adj._indices()
    #     # the input data excludes self-connections but [gat] uses them. So we need to add them
    #     edge_index_ = self.add_self_connections_to_edge_index(edge_index, n)
    #     # logger.info("edge_index_ Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     h_prime = self.lin(h).view(n, self.n_head, self.f_out)  # (N, F_IN)*(F_IN, K*F_OUT) -> (N, K*F_OUT) -> (N, K, F_OUT)
    #     # logger.info("h_prime Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     # edge_index_[0] = source; edge_index_[1] = target
    #     h_src = (h_prime[edge_index_[0]] * self.a_src).sum(dim=2)  # ((N, K, F_OUT) * (1, K, F_OUT)).sum(2) -> (N, K)
    #     h_trg = (h_prime[edge_index_[0]] * self.a_trg).sum(dim=2)
    #     # logger.info("h_src/h_trg Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     # compute raw attention coefficients `e_{ij}` in paper [gat, p.3]
    #     e = h_src + h_trg  # (E, K)
    #     e = self.leaky_relu(e)
    #     # logger.info("leaky_relu Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     # apply softmax normalization over all source nodes per target node
    #     e = e - e.max()   # trick to improve numerical stability before computing exponents (for softmax)
    #     exp_e = e.exp()  # = unnormalized attention for each pair self.edge_pair[0]->self.edge_pair[1]
    #     # logger.info("e.exp Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     trg_normalization = torch.zeros(size=(n, self.n_head), dtype=h.dtype, device=h.device)  # tensor with normalizing constants for each target node
    #     # logger.info("trg_norm created Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     index = self.explicit_broadcast(edge_index_[1], exp_e)
    #     # logger.info("explicit_broadcast Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     trg_normalization.scatter_add(dim=0, index=index, src=exp_e)   # index1:[E, K], exp_e:[E, K] -> [N, K]
    #     # logger.info("trg_norm scatter_add_ Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     # normalized attention coefficients `alpha_{ij}` in paper [gat, p.3]
    #     alpha = exp_e / (trg_normalization[edge_index_[1]] + 1e-10)  # (E, K), s.t. for a given target, the sum of the sources = 1.
    #     # logger.info("normalized Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     # alpha = dropout(alpha)
    #     src = h_prime[edge_index_[0]]  # (E, K, F_OUT)
    #     # logger.info("src Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     src = src * alpha.unsqueeze(-1)
    #     # logger.info("src* Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     out = torch.zeros(size=(n, self.n_head, self.f_out), dtype=h.dtype, device=h.device)
    #     # logger.info("out Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     index = self.explicit_broadcast(edge_index_[1], src)
    #     # logger.info("index Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
        
    #     out = out.scatter_add(0, index, src)  # (N, K, F_OUT)  # h double prime
    #     # logger.info("out scatter_add Allocated: {torch.cuda.memory_reserved(device)/1024**3}")

    #     out = out.permute(1,0,2)

    #     if self.bias is not None:
    #         return out + self.bias
    #     else:
    #         return out

    # NOTE: Another Method, GPU Mem not enough, but can solve following problem,
    # RuntimeError: The backward pass for this operation requires the 'self' tensor to be strided, but a sparse tensor was given instead. Please either use a strided tensor or set requires_grad=False for 'self'
    # def forward2(self, x: torch.Tensor, adj: torch.Tensor):
    #     N = x.size(0)
    #     h = torch.matmul(x.unsqueeze(0), self.w)
    #     edge = adj._indices()
        
    #     output_l = []
    #     for head_idx in range(self.n_head):
    #         h_, a_ = h[head_idx], self.a[head_idx] # (N, f_out)

    #         # Self-attention on the nodes - Shared attention mechanism
    #         edge_h = torch.cat((h_[edge[0,:], :], h_[edge[1,:], :]), dim=1).t()
    #         # edge: 2*D x E

    #         edge_e = torch.exp(-self.leaky_relu(a_.mm(edge_h).squeeze()))
    #         # edge_e: E

    #         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device='cuda'))
    #         # e_rowsum: N x 1

    #         edge_e = self.dropout(edge_e)
    #         # edge_e: E

    #         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h_)
    #         # h_prime: N x out

    #         h_prime = h_prime.div(e_rowsum)
    #         # h_prime: N x out

    #         output_l.append(h_prime.unsqueeze(0))
    #     ret = torch.cat(output_l, dim=0) # (n_head, N, f_out)

    #     if self.bias is not None:
    #         return ret + self.bias
    #     else:
    #         return ret


class AdditiveAttention(nn.Module):
    def __init__(self, d, d1, d2) -> None:
        super().__init__()

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

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
        nb_node_kind = type_aware_emb.shape[1]
        feature = feature.unsqueeze(1).repeat(1, nb_node_kind, 1) # (N, |Rs|, D)
        # logger.info(feature.shape)

        q = self.tanh(self.w1(feature)+self.w2(type_aware_emb)) # (N, |Rs|, D'')
        q = self.m(q) # (N, |Rs|, 1)
        beta = self.softmax(q.squeeze(2)) # (N, |Rs|)
        type_fusion_emb = torch.bmm(beta.unsqueeze(1), type_aware_emb) # (N, 1, D')

        return type_fusion_emb


class HeterogeneousGraphAttention(nn.Module):
    def __init__(self, n_user, nb_nodes=3, n_units=[100, 16], n_heads=[3], attn_dropout=0.0, dropout=0.1, d2=128, gpu_device_ids=[]) -> None:
        """
        Args:
            nb_nodes(default=3): 异质图节点类型数量
            gpu_device_ids(List[int], default=[]): 采用Model Parallel方法, 主要使多头注意力可以在不同GPU上运行
                [0]永远是model.device
        """
        super().__init__()

        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.gpu_device_ids = gpu_device_ids

        self.d = n_units[0]
        self.d1 = n_units[1]
        self.d2 = d2
        self.n_user = n_user

        self.layer_stack = nn.ModuleList()
        for hidx in range(nb_nodes):
            layer_stack = nn.ModuleList()
            for i in range(self.n_layer):
                # consider multi head from last layer
                f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
                layer_stack.append(
                    SparseMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                        f_out=n_units[i + 1], attn_dropout=attn_dropout)
                )
            if self.gpu_device_ids:
                device = self.gpu_device_ids[hidx%len(self.gpu_device_ids)]
                layer_stack = layer_stack.to(f'cuda:{device}')
            self.layer_stack.append(layer_stack)
        self.additive_attention = AdditiveAttention(d=self.d, d1=self.d1, d2=self.d2)
        if self.gpu_device_ids:
            self.additive_attention = self.additive_attention.to('cuda:1')
    
    def forward(self, h, hadj):
        """
        Args:
            x: initial feature of all main nodes, shape is (N, D)
            hadj: ~, shape is (|Rs|, N, N)
        Return:
            type_aware_emb, type_fusion_emb: |Nu, |Rs|+1, D'|
        """
        assert len(hadj) == len(self.layer_stack)
        h_device = h.device
        # logger.info(f"Before* Allocated: {torch.cuda.memory_reserved(h_device)/1024**3}")
        h_embs = []
        for hidx, adj in enumerate(hadj):
            layer_stack = self.layer_stack[hidx]
            if self.gpu_device_ids:
                device = self.gpu_device_ids[hidx%len(self.gpu_device_ids)]
                h = h.to(f'cuda:{device}')
                adj = adj.to(f'cuda:{device}')
            for i, gat_layer in enumerate(layer_stack):
                x = gat_layer(h, adj) # (n_head, n, f_out)
                if i + 1 == self.n_layer:
                    x = x.mean(dim=0)
                else:
                    x = F.elu(x.view(x.shape[1], -1))
                    x = F.dropout(x, self.dropout, training=self.training)
                # logger.info(f"After GAT-Layer Allocated: {torch.cuda.memory_reserved(device)/1024**3}")
            h_embs.append(x[:self.n_user].unsqueeze(1)) # (Nu, 1, f_out=n_units[-1]*n_heads[-1])
        if self.gpu_device_ids:
            h = h.to(h_device)
            h_embs = [elem.to(h_device) for elem in h_embs]
        type_aware_emb = torch.cat(h_embs, dim=1) # (Nu, |Rs|, D')
        type_fusion_emb = self.additive_attention(h[:self.n_user], type_aware_emb) # (Nu, 1, D')
        return torch.cat((type_fusion_emb, type_aware_emb), dim=1)
