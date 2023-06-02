# 1. create (center_word, neighbor_words) tuples from each hashtag

import logging
import datetime

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter

import re

class SG_ModelInput:
    def __init__(self, center_word, positvie_words, negative_words):
        self.center_word = center_word
        self.pos_words = positvie_words
        self.neg_words = negative_words
    
    def __str__(self):
        return f"center_word: {self.center_word}, pos_words:{self.pos_words}, neg_words[:len(pos_words)]: {self.neg_words[:len(self.pos_words)]}"

import json

sentence_list = []
graphid_dict = dict()

# Virality2013
diffusion_dict_filepath = "/root/TR-pptusn/DeepInf-preprocess/output/data/virality2013-new-output/ce_init_diffusion_dict.npy"
model_name = "virality2013"

# Munmun
# diffusion_dict_filepath = "/root/TR-pptusn/DeepInf-preprocess/output/data/munmun-new-output/ce_init_diffusion_dict.npy"
# model_name = "munmun"

# Higgs
# diffusion_dict_filepath = "/root/TR-pptusn/DeepInf-preprocess/output/data/higgs-new-output/ce_init_diffusion_dict.npy"
# model_name = "higgs"

with open(diffusion_dict_filepath, 'r') as f:
    for line in f:
        if line[0] == '{':
            line = line.replace("'", "\"")
            graphid_dict = json.loads(line)
            continue
        if line[0] != '[':
            continue
        cascade = re.sub('\s|\t|\n|"|\'', '', line[1:-2])
        sentence = cascade.split(',')
        sentence_list.append(sentence)

graph_vcount = len(graphid_dict)
word_vocabulary_dict = dict( (str(idx), idx) for idx in range(graph_vcount) )

logger.info(f"Read from File {diffusion_dict_filepath}, Containing {len(sentence_list)} Cascades and {graph_vcount} Nodes with a vocab")

import numpy as np

user_count = dict((i,0) for i in range(graph_vcount))
for sentence in sentence_list:
    for word in sentence:
        user_count[int(word)] += 1

user_count = np.array(list(user_count.values())) ** 0.75
user_count_normalized = user_count / sum(user_count)

# neg_sample_distribution = np.round(user_count * 1e8)
# neg_sample_pool = []
# for neg_word_idx, neg_word_count in enumerate(neg_sample_distribution):
#     neg_sample_pool += [neg_word_idx] * int(neg_word_count)

import numpy as np
import copy

window_size = 10
negative_sampling_num_per_word = 20 # other possible values: 25 or 15

def get_sgmodel_item(center_word_idx, sentence):
    sentence_len = len(sentence)
    # start_idx, end_idx = max(center_word_idx - window_size, 0), min(center_word_idx + window_size, sentence_len-1)
    pos_words_idx = np.concatenate([
        np.arange(center_word_idx - window_size, center_word_idx, 1), 
        np.arange(center_word_idx+1, center_word_idx + window_size + 1, 1)
    ])
    pos_words = [int(sentence[pw_idx]) if pw_idx >= 0 and pw_idx < sentence_len else 0 for pw_idx in pos_words_idx]

    # mask = np.ones(sentence_len)
    mask = copy.deepcopy(user_count_normalized)
    mask_pos_words_idx = list(filter(lambda item: item >= 0 and item < sentence_len, pos_words_idx))
    mask[mask_pos_words_idx], mask[center_word_idx] = 0, 0
    mask = mask / mask.sum()
    neg_words = np.random.choice([i for i in range(graph_vcount)], size=negative_sampling_num_per_word * len(pos_words_idx), p=mask)
    
    return SG_ModelInput(
        center_word=int(sentence[center_word_idx]),
        # positvie_words=np.array(sentence)[pos_words_idx].tolist(),
        positvie_words=pos_words,
        negative_words=neg_words
    )

# sample_item = get_sgmodel_item(4, sentence_list[34])

import pickle

sgmodel_item_list = []
for idx, sentence in enumerate(sentence_list):
    # print(f"{idx}")
    if idx % 10 == 0:
        logger.info(f"idx: {idx}")
    for word_idx in range(len(sentence)):
        sgmodel_item_list.append(get_sgmodel_item(word_idx, sentence))

logger.info(f"sgmodel_item_list length: {len(sgmodel_item_list)}")
with open(f"data/sgmodel-item-list/{model_name}-sgmodel_item_list", "wb") as f:
    pickle.dump(sgmodel_item_list, f)


# 2. SkipGram模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F

# import logging
# logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.in_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.out_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    
    def forward(self, center_word_idx, pos_words_idx, neg_words_idx):
        if use_cuda:
            center_word_idx = center_word_idx.cuda()
            pos_words_idx, neg_words_idx = pos_words_idx.cuda(), neg_words_idx.cuda()
        
        center_word_idx = center_word_idx.squeeze()
        
        cw_emb = self.in_embedding(center_word_idx)
        pos_emb, neg_emb = self.out_embedding(pos_words_idx), self.out_embedding(neg_words_idx)

        cw_emb = cw_emb.unsqueeze(2)
        # print(f"cw_emb: {cw_emb.size()}, pos_emb: {pos_emb.size()}, neg_emb: {neg_emb.size()}")

        pos_score = torch.bmm(pos_emb, cw_emb).squeeze(2)
        pos_score = -torch.sum(F.logsigmoid(pos_score), dim=1)

        neg_score = torch.bmm(neg_emb, -cw_emb).squeeze(2)
        neg_score = -torch.sum(F.logsigmoid(neg_score), dim=1)
        # print(f"pos_score: {pos_score.mean()}, neg_score: {neg_score.mean()}")

        return pos_score + neg_score
    
    def get_in_embedding(self):
        return self.in_embedding.weight.detach()

# 3. Train and Test
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

vocab_size = graph_vcount
# batch_size = 256
batch_size = 256
epochs = 5
lr = 0.003
hidden_size = 128

class SGModelDataSet(Dataset):
    def __init__(self, sgmodel_item_list):
        self.sgmodel_item_list = sgmodel_item_list
    
    def __len__(self):
        return len(self.sgmodel_item_list)
    
    def __getitem__(self, idx):
        cw, pws, nws = [int(self.sgmodel_item_list[idx].center_word)], \
            [int(elem) for elem in self.sgmodel_item_list[idx].pos_words], \
            [int(elem) for elem in self.sgmodel_item_list[idx].neg_words]
        return torch.LongTensor(cw), torch.LongTensor(pws), torch.LongTensor(nws)

data_loader = DataLoader(
    SGModelDataSet(sgmodel_item_list=sgmodel_item_list),
    batch_size=batch_size,
    # shuffle=True
)
model = SkipGramModel(vocab_size=vocab_size, hidden_size=hidden_size)
if use_cuda:
    model = model.cuda()
optimizer = optim.Adagrad(params=model.parameters(), lr=lr)

for epoch_idx in range(epochs):
    for batch_idx, batch in enumerate(data_loader):
        center_word, pos_words, neg_words = batch
        if use_cuda:
            center_word, pos_words, neg_words = center_word.cuda(), pos_words.cuda(), neg_words.cuda()
        
        optimizer.zero_grad()
        output = model(center_word, pos_words, neg_words)
        loss = output.mean()
        loss.backward()
        optimizer.step()

        # if batch_idx and batch_idx % 100 == 0:
        #     print(f"epoch_idx: {epoch_idx}, batch_idx: {batch_idx}, loss: {loss.item()}")

emb = model.get_in_embedding().cpu().numpy()
with open(f"data/cascade-embedding/{model_name}-cascade_embedding.npy", "w") as f:
    f.write(f"{vocab_size} {hidden_size}\n")
    for word_idx in range(graph_vcount):
        emb_idx = emb[word_idx].tolist()
        f.write(f"{word_idx} {(emb_idx)}\n")
