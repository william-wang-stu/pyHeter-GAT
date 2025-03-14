import pickle
import numpy as np
# import pandas as pd
import os
import re
from openai import OpenAI
import httpx
import json
import random
import regex
import glob
import requests
# from bs4 import BeautifulSoup
from typing import Optional
import torch
from torch_geometric.data import Data

def save_pickle(obj, filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p','.data']:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    elif ext == '.npy':
        if not isinstance(obj, np.ndarray):
            obj = np.array(obj)
        np.save(filename, obj)
    else:
        pass # raise Error

def load_pickle(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p','.data']:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    elif ext == '.npy':
        return np.load(filename)
    else:
        return None # raise Error

def analyse_distribution(data):
    for i in range(10):
        print(np.percentile(data, i*10))

# 1. 抽取话题
### start of 1 ###
# result_filepath = "/remote-home/share/dmb_nas/ppx/result_twitter.jsonl"

# [1] Politics: Mentions the involvement of the President of the United States and USAID in political actions related to socialism and corruption.
# regex_pattern = r"\[(\d+)\] ([\w\s]+): ([\w\s]+)\."
# regex_pattern = r'\[1\] (.*?): (.*?) \("(.*?)"\) \| Confidence: (\d+)'

# miss_cnt = 0
# generated_topics = {}

# with open(result_filepath, 'r') as f:
#     for line in f:
#         # resp = line
#         resp = json.loads(line)
#         # print(json.dumps(resp, indent=4, ensure_ascii=False))

#         match = re.match(regex_pattern, resp['response'])
#         if match:
#             category = match.group(1)
#             # description = match.group(2)
#             # quote = match.group(3)
#             confidence = match.group(4)

#             # NOTE: confidence is not used
#             if category not in generated_topics: generated_topics[category] = {"count": 0}
#             generated_topics[category]["count"] += 1
#         else:
#             miss_cnt += 1

# print(len(generated_topics))
# print(sum([v["count"] for k,v in generated_topics.items()]))

# wiki_kb_filepath = "/root/pyHeter-GAT/preprocess-llm/output/wikipedia_class.txt"

# first_topics = []
# second_topics = []
# st_ft_mapping = {}
# last_ft = None
# with open(wiki_kb_filepath, 'r') as f:
#     for line in f:
#         if line[0] == ' ':
#             st_ft_mapping[line.strip()] = last_ft
#             second_topics.append(line.strip())
#         else:
#             last_ft = line.strip()
#             first_topics.append(line.strip())

# # add first topics
# for ft in first_topics:
#     st_ft_mapping[ft] = ft

# print(first_topics.keys())

# # 5-2-2. Prepare sBert encoding
# from sentence_transformers import SentenceTransformer, util

# sbert = SentenceTransformer("/remote-home/share/dmb_nas/wangzejian/LLM_GNN/all-MiniLM-L6-v2")

# f_topic_mp = {}
# f_topic_embs = sbert.encode(first_topics)
# for tp, emb in zip(first_topics, f_topic_embs):
#     f_topic_mp[tp] = emb

# s_topic_mp = {}
# s_topic_embs = sbert.encode(second_topics)
# for tp, emb in zip(second_topics, s_topic_embs):
#     s_topic_mp[tp] = emb

# # 5-2-3. Find the most related wikipedia topics for TopicGPT-generated results
# gts = list(generated_topics.keys())

# gt_ft_mapping = {}
# gt_embs = sbert.encode(gts)
# for emb, tp in zip(gt_embs, gts):
#     max_sim = -1
#     max_tp = ""
#     for f_tp, f_emb in f_topic_mp.items():
#         sim = util.pytorch_cos_sim(emb, f_emb)
#         if sim > max_sim:
#             max_sim = sim
#             max_tp = f_tp

#     for s_tp, s_emb in s_topic_mp.items():
#         sim = util.pytorch_cos_sim(emb, s_emb)
#         if sim > max_sim:
#             max_sim = sim
#             max_tp = s_tp
#     gt_ft_mapping[tp] = st_ft_mapping[max_tp]

# save_pickle(gt_ft_mapping, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm_ppx/gt_ft_mapping.data")

### end of 1 ###

# 2. 生成话题通道

# 4-1. Prepare input from TopicGPT
result_filepath = "/remote-home/share/dmb_nas/ppx/result_twitter.jsonl"
regex_pattern = r'\[1\] (.*?): (.*?) \("(.*?)"\) \| Confidence: (\d+)'

miss_cnt = 0
results = []

with open(result_filepath, 'r') as f:
    for line in f:
        # resp = line
        resp = json.loads(line)
        # print(json.dumps(resp, indent=4, ensure_ascii=False))

        match = re.match(regex_pattern, resp['response'])
        if match:
            category = match.group(1)
            # description = match.group(2)
            # quote = match.group(3)
            confidence = match.group(4)

            # NOTE: confidence is not used
            results.append([category, int(resp['id']), int(resp['ts'])])
        else:
            miss_cnt += 1
print(len(results), miss_cnt)

# 4-1-2. Prepare topic to interest mappings
gt_ft_mapping = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm_ppx/gt_ft_mapping.data")

# 4-2. generate simedges from interest-aware cascades
def generate_interest_cascades(data_dict: list, remove: bool = False):
    interest_cascades = {}
    for elem in data_dict:
        if gt_ft_mapping[elem[0]] not in interest_cascades: interest_cascades[gt_ft_mapping[elem[0]]] = []
        interest_cascades[gt_ft_mapping[elem[0]]].append((elem[1], int(elem[2])))
    print(len(interest_cascades))
    
    # remove minor interests
    interest_cascades_cp = interest_cascades.copy()
    for t, cascades in interest_cascades_cp.items():
        if len(cascades) < 10:
            interest_cascades.pop(t)
    print(len(interest_cascades))

    # sort by timestamp
    for t in interest_cascades:
        interest_cascades[t] = sorted(interest_cascades[t], key=lambda x: x[1])
    return interest_cascades

def generate_simedges(interest_cascades: dict, time_distance: int = 3600 * 24 * 30):
    simedges = {}
    for interest, cascades in interest_cascades.items():
        simedges[interest] = []
        for i in range(len(cascades)-1):
            for j in range(i, len(cascades)-1):
                if cascades[j][1] - cascades[i][1] < time_distance and \
                    cascades[j][0] != cascades[i][0]:
                # if cascades[j][0] != cascades[i][0]:
                    simedges[interest].append((cascades[i][0], cascades[j][0]))
        # for i in range(len(cascades)-1):
        #     for j in range(max(0,i-1-window_size),min(i+1+window_size,len(cascades))): # (i-ws-1<-i->i+ws+1)
        #         if cascades[i][0] != cascades[j][0]:
        #             simedges[interest].append((cascades[i][0],cascades[j][0]))
    
    # remove abundant edges
    for interest, edges in simedges.items():
        simedges[interest] = list(set(edges))
    return simedges

# 4-3. convert simedges to graph
def convert_to_graph(simedges: dict, user_size: int, originial_edges: Optional[list] = None):
    if originial_edges:
        for interest, edges in simedges.items():
            simedges[interest] = list(set(edges + originial_edges))
    
    # add self-loop edges
    for interest, edges in simedges.items():
        edges += [(u,u) for u in range(user_size)]
        simedges[interest] = list(set(edges))
    
    # convert to graph
    graph_d = {}
    for interest, edges in simedges.items():
        edges = list(zip(*edges))
        edges_t = torch.LongTensor(edges)
        weight_t = torch.FloatTensor([1]*edges_t.size(1))
        graph_d[interest] = Data(edge_index=edges_t, edge_weight=weight_t)
    
    return graph_d

def sort_dict(m: dict) -> dict:
    return dict(sorted(m.items(), key=lambda x: x[1], reverse=True))

days = 3
time_distance = 3600 * 24 * days
save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm_ppx/topic_diffusion_graph_full_windowsize{td}.data"
save_filepath = save_filepath.format(td=days)
print(save_filepath)

interest_cascades = generate_interest_cascades(results, remove=True)

simedges = generate_simedges(interest_cascades, time_distance)

# select top20 interests
len_mp = {k:len(v) for k,v in simedges.items()}
for k in list(sort_dict(len_mp).keys())[20:]:
    # print(k, len(simedges[k]))
    simedges.pop(k)

u2idx = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/u2idx.data")
original_edges = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/edges.data")
graph_d = convert_to_graph(simedges, user_size=len(u2idx), originial_edges=original_edges)
for key, graph in graph_d.items():
    print(graph.edge_index.size())
save_pickle(graph_d, save_filepath)
