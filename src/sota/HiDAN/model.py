import numpy as np

# def ill_cal(pred, sl):
#     nll = 0
#     cur_pos = 0
#     for i in range(len(sl)):
#         length = sl[i]
#         cas_nll = pred[cur_pos : cur_pos+length]
#         cur_pos += length
#         nll += (np.sum(cas_nll)/float(length))
#     return nll

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiDAN(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(HiDAN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

    def forward(self, cas_emb, cas_mask, time_weight):
        # cas_emb:[b,n,d]  cas_mask:[b,n,1]
        cas_encoding = self.user2user(cas_emb, cas_mask)
        return self.user2cas(cas_encoding, cas_mask, time_weight)

    def user2user(self, cas_emb, cas_mask):
        bs, sl, _ = cas_emb.size()
        col = torch.arange(sl, device=cas_emb.device).unsqueeze(0).repeat(sl, 1)
        row = torch.arange(sl, device=cas_emb.device).unsqueeze(1).repeat(1, sl)
        direction_mask = (row > col).to(torch.int32)
        direction_mask_tile = direction_mask.unsqueeze(0).repeat(bs, 1, 1)
        length_mask_tile = cas_mask.squeeze(-1).unsqueeze(1).repeat(1, sl, 1)
        attention_mask = (direction_mask_tile & length_mask_tile).float()

        cas_hidden = self.dense(cas_emb, self.hidden_size, F.elu) * cas_mask

        head = self.dense(cas_hidden, self.hidden_size, lambda x: x)
        tail = self.dense(cas_hidden, self.hidden_size, lambda x: x)

        matching_logit = torch.matmul(head, tail.transpose(1, 2)) + (1 - attention_mask) * (-1e30)
        attention_score = F.softmax(matching_logit, dim=-1) * attention_mask
        depend_emb = torch.matmul(attention_score, cas_hidden)

        fusion_gate = self.dense(torch.cat([cas_hidden, depend_emb], dim=2), self.hidden_size, torch.sigmoid)
        return (fusion_gate * cas_hidden + (1 - fusion_gate) * depend_emb) * cas_mask

    def user2cas(self, cas_encoding, cas_mask, time_weight):
        map1 = self.dense(cas_encoding, self.hidden_size, F.elu)
        time_influence = self.dense(time_weight, self.hidden_size, F.elu)
        map2 = self.dense(map1 * time_influence, 1, lambda x: x)
        attention_score = F.softmax(map2 + (-1e30) * (1 - cas_mask), dim=1) * cas_mask
        # return torch.sum(attention_score * cas_encoding, dim=1)
        return attention_score * cas_encoding

    def dense(self, input, out_size, activation):
        input_shape = input.size()
        input_flatten = input.view(-1, input_shape[-1])
        weight = nn.Parameter(torch.randn(input_shape[-1], out_size, device=input.device))
        bias = nn.Parameter(torch.zeros(out_size, device=input.device))
        out = torch.matmul(input_flatten, weight) + bias
        out = activation(out).view(*input_shape[:-1], out_size)
        return F.dropout(out, p=1 - self.dropout, training=self.training)


class HiDANModel(nn.Module):
    def __init__(self, config):
        super(HiDANModel, self).__init__()
        self.num_nodes = config.num_nodes
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.learning_rate = config.learning_rate
        self.l2_weight = config.l2_weight
        self.train_dropout = config.dropout
        self.n_time_interval = config.n_time_interval
        self.optimizer = config.optimizer

        self.embedding = nn.Embedding(self.num_nodes, self.embedding_size)
        self.time_lambda = nn.Parameter(torch.randn(self.n_time_interval + 1, self.hidden_size))
        self.hidan_layer = HiDAN(self.hidden_size, self.train_dropout)

    def forward(self, cas, time_interval_index):
        cas = cas[:, :-1]
        time_interval_index = time_interval_index[:, :-1]
        cas_length = cas.ne(0).sum(dim=1)
        cas_mask = (cas_length.unsqueeze(1).unsqueeze(2).float() > torch.arange(cas.size(1), device=cas.device).unsqueeze(0).unsqueeze(2)).to(torch.int32)

        cas_emb = self.embedding(cas)
        time_weight = self.time_lambda[time_interval_index]

        hidan_output = self.hidan_layer(cas_emb, cas_mask, time_weight)
        logits = self.dense(hidan_output, self.num_nodes, lambda x: x)
        # loss = F.cross_entropy(logits, labels)

        # for param in self.parameters():
        #     loss += self.l2_weight * torch.norm(param, p=2)

        return logits.view(-1, self.num_nodes)

    def dense(self, input, out_size, activation):
        input_shape = input.size()
        input_flatten = input.view(-1, input_shape[-1])
        weight = nn.Parameter(torch.randn(input_shape[-1], out_size, device=input.device))
        bias = nn.Parameter(torch.zeros(out_size, device=input.device))
        out = torch.matmul(input_flatten, weight) + bias
        out = activation(out).view(*input_shape[:-1], out_size)
        return out

# class Model(object):
#     def __init__(self, config):
#         self.num_nodes = config.num_nodes
#         self.hidden_size = config.hidden_size
#         self.embedding_size = config.embedding_size
#         self.learning_rate = config.learning_rate
#         self.l2_weight = config.l2_weight
#         self.train_dropout = config.dropout
#         self.n_time_interval = config.n_time_interval
#         self.optimizer = config.optimizer


#     def build_model(self):
#         with tf.compat.v1.variable_scope("model",initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")) as scope:
#             self.cas = tf.compat.v1.placeholder(tf.int32, [None, None])                    # (b,n)

#             self.cas_length= tf.reduce_sum(input_tensor=tf.sign(self.cas),axis=1)
#             self.cas_mask = tf.expand_dims(tf.sequence_mask(self.cas_length, tf.shape(input=self.cas)[1], tf.float32), -1)

#             self.dropout = tf.compat.v1.placeholder(tf.float32)
#             self.labels = tf.compat.v1.placeholder(tf.int32, [None])                          # (b,)

#             self.time_interval_index = tf.compat.v1.placeholder(tf.int32, [None, None])       # (b,n)

#             self.num_cas = tf.compat.v1.placeholder(tf.float32)

#             with tf.device("/cpu:0"):
#                 self.embedding = tf.compat.v1.get_variable(
#                     "embedding", [self.num_nodes,
#                         self.embedding_size], dtype=tf.float32)
#                 self.cas_emb = tf.nn.embedding_lookup(params=self.embedding, ids=self.cas)      # (b,n,l)

#                 self.time_lambda = tf.compat.v1.get_variable('time_lambda', [self.n_time_interval+1, self.hidden_size], dtype=tf.float32) #,
#                 self.time_weight = tf.nn.embedding_lookup(params=self.time_lambda, ids=self.time_interval_index)

#             with tf.compat.v1.variable_scope("hidan") as scope:
#                 self.hidan = hidan(self.cas_emb, self.cas_mask, self.time_weight, self.hidden_size, self.dropout)

#             with tf.compat.v1.variable_scope("loss"):
#                 l0 = self.hidan
#                 self.logits = dense(l0, self.num_nodes, tf.identity, 1.0, 'logits')
#                 self.nll = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(tf.one_hot(self.labels, self.num_nodes, dtype=tf.float32)), logits=self.logits)
#                 self.loss = tf.reduce_mean(input_tensor=self.nll,axis=-1)
#                 for v in tf.compat.v1.trainable_variables():
#                     self.loss += self.l2_weight * tf.nn.l2_loss(v)
#                 if self.optimizer == 'adaelta':
#                     self.train_op = tf.compat.v1.train.AdadeltaOptimizer(self.learning_rate, rho=0.999).minimize(self.loss)
#                 else:
#                     self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=0.99).minimize(self.loss)

    # def train_batch(self, sess, batch_data):
    #     cas, next_user, time_interval_index, seq_len = batch_data
    #     feed = {self.cas: cas,
    #             self.labels: next_user,
    #             self.dropout: self.train_dropout,
    #             self.time_interval_index: time_interval_index,
    #             self.num_cas: len(seq_len)
    #            }
    #     _, _, nll = sess.run([self.train_op, self.loss, self.nll], feed_dict = feed)
    #     batch_nll = np.sum(nll)
    #     return batch_nll

    # def test_batch(self, sess, batch_test):
    #     cas, next_user, time_interval_index, seq_len = batch_test
    #     feed = {self.cas: cas,
    #             self.labels: next_user,
    #             self.time_interval_index: time_interval_index,
    #             self.dropout: 1.0
    #            }
    #     logits, nll = sess.run([self.logits, self.nll], feed_dict = feed)
    #     # batch_rr = mrr_cal(logits, next_user, seq_len)
    #     # mrr, macc1, macc5, macc10, macc50, macc100 = rank_eval(logits, next_user, seq_len)
    #     mrr, macc10, macc50, macc100, mmap10, mmap50, mmap100 = rank_eval(logits, next_user, seq_len)
    #     batch_cll = np.sum(nll)
    #     batch_ill = ill_cal(nll, seq_len)
    #     return batch_cll, batch_ill, mrr, macc10, macc50, macc100, mmap10, mmap50, mmap100
