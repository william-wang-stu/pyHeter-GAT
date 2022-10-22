from torch import nn
from torch import Tensor
from scipy.sparse import csr_matrix
from log import logger

class TypeAwareAlignmentLoss(nn.Module):
    def __init__(self, epsilon=3, omega=0.4, r=[0.2,0.2,0.2]):
        """
        """
        super().__init__()

        self.epsilon = epsilon
        self.omega = omega
        self.r = r
        self.loss_fn = nn.L1Loss(reduction="sum")

    # def get_loss_matrix(self, emb_s: Tensor, emb_t: Tensor):
    #     """
    #     Args:
    #         emb_s, emb_t: shape is (N, |Rs|, D')
    #     """
    #     # |Rs| in Gs and Gt
    #     Ns, Nt = emb_s.shape[0], emb_t.shape[0]
    #     NRs, NRt = emb_s.shape[1], emb_t.shape[1]

    #     if NRs != NRt:
    #         print("Error: type of nodes in Gs=%d and Gt=%d are different", NRs, NRt)
    #         return
        
    #     def loss(emb1, emb2):
    #         """
    #         Args:
    #             emb1, emb2: shape is (|Rs|, D')
    #         """
    #         # ls = 0
    #         # len = emb1.shape[0]
    #         # for i in range(len):
    #         #     param = self.r[i-1] if i else self.omega
    #         #     ls += param * self.loss_fn(emb1[i], emb2[i])
    #         # return ls

    #         len = emb1.shape[0]
    #         param_mat = [self.omega, *self.r]
    #         loss_mat = [self.loss_fn(emb1[i], emb2[i]) for i in range(len)]
    #         return np.sum(np.multiply(param_mat, loss_mat))

    #     Z = []
    #     for x in np.arange(Ns):
    #         # if x % 100 == 0:
    #         #     print(x)
    #         for y in np.arange(Nt):
    #             Z.append(loss(emb_s[x], emb_t[y]))
    #     Z = np.array(Z).reshape(Ns, -1)
    #     return torch.from_numpy(Z)

    def forward(self, anchor_link_matrix: csr_matrix, emb_s: Tensor, emb_t: Tensor):
        """
        Args:
            anchor_link_matrix: (i,j) indicates whether i_s and j_t is an anchor link
            # ta_s, ta_t: type_aware_emb in Gs and Gt
            # tf_s, tf_t: type_fusion_emb in Gs and Gt
            emb_s, emb_t: {type_fusion_emb, type_aware_emb} in Gs and Gt
        Return:
            Loss(i_s,j_t)= \
                w*d(tf_s(i), tf_t(j))+\Sigma{r*d(ta_s(i), ta_t(j))},    (i_s, j_t) is anchor link,
                -w*d(tf_s(i), tf_t(j))-\Sigma{r*d(ta_s(i), ta_t(j))},   (i_s, j_t) is not anchor link,
            Loss = \Sigma_{(i_s,j_t) is anchor link} \Sigma_{(i_s',j_t') is not anchor link} { Loss(i_s,j_t)+Loss(i_s',j_t') }
                 = (ALM, (I-ALM)) @ (|B|, N^2-|B|)^T @ Loss-Matrix
        """
        # logger.info(f"epsilon={self.epsilon}, omega={self.omega}, r={self.r}")
        # logger.info(f"emb_s device={emb_s.device}, emb_t device={emb_t.device}")
        Ns, Nt = emb_s.shape[0], emb_t.shape[0]
        NRs, NRt = emb_s.shape[1], emb_t.shape[1]
        d1 = emb_s.shape[2]
        # device = emb_s.device
        # logger.info(f"Ns={Ns}, Nt={Nt}, NRs={NRs}, NRt={NRt}, d1={d1}, device={device}")

        ALM = anchor_link_matrix
        nb_B = len(ALM.row)
        nb_notB = Ns*Nt - nb_B

        if NRs != NRt:
            logger.info(f"Error: type of nodes in Gs={NRs} and Gt={NRt} are different")

        # Build Matched Pair Map
        findj = {}
        for row, col in zip(ALM.row, ALM.col):
            findj[row] = col

        # Calc B and B'
        ret = 0
        for k in range(NRs):
            param = self.r[k-1] if k else self.omega
            loss_ALM = 0
            loss_notALM = 0
            for i in range(Ns):
                # if i % 1000 == 0:
                #     print(f"Allocated idx={i}: {torch.cuda.memory_allocated(device)/1024**3}")
                #     print(f"Cached idx={i}: {torch.cuda.memory_reserved(device)/1024**3}")
                # calc B
                if i in findj:
                    j = findj[i]
                    loss_ALM += self.loss_fn(emb_s[i,k], emb_t[j,k])
                # calc B'
                s = emb_s[i,k].unsqueeze(0).repeat(Nt,1) #(Nt, D')
                t = emb_t[:,k] #(Nt, D')
                # if i in findj:
                #     j = findj[i]
                #     # mask matched jth row with s1[i,k]
                #     # so that L1Loss(t[j], s[i]) = 0
                #     # NOTE: NO in-place op
                #     # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [16]], which is output 0 of AsStridedBackward0, is at version 13128; expected version 13127 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
                #     t[j] = s[0]
                # NOTE: Use .item() Here
                # therefore those GPU Mem allocated for variable s would be freed, 
                # since pytorch wouldn't keep the computation graph
                # Sad!!! But NO backward
                loss_notALM += self.loss_fn(s, t)
            # print(f"param={param}, loss_ALM={loss_ALM}, loss_notALM={loss_notALM}")
            loss_notALM -= loss_ALM
            loss = loss_ALM*nb_notB + (self.epsilon*nb_notB-loss_notALM)*nb_B 
            # print(f"total loss={loss/d1}")
            ret += param*loss/d1
        
        # Avg on all (i,j) Pairs
        ret = ret / (Ns*Nt)

        return ret
