import pickle
import numpy as np
# import pandas as pd
import os
from openai import OpenAI
import httpx
import json
import random
import regex
import glob
import requests
from bs4 import BeautifulSoup
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

# 1-1. 降采样: 将推文按照用户粒度聚合

cascade_dict = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/cascades.data")

user_tweets_dict = {}
for tag, cascades in cascade_dict.items():
    for user, ts, tweet, bertopic_label in zip(cascades['user'], cascades['ts'], cascades['content'], cascades['label']):
        if user not in user_tweets_dict:
            user_tweets_dict[user] = []
        user_tweets_dict[user].append({
            "tag": tag,
            "bertopic_label": str(bertopic_label),
            "user": user,
            "ts": ts,
            "text": tweet,
        })

# # 查看Weibo数据集的推文数据量级
# # 总共73w条推文, 不过用户总量是4971, 按照每个用户粒度降采样应该能达到不错的覆盖率

# cascade_dict = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/cascades.data")

# user_tweets_dict = {}
# for tag, cascades in cascade_dict.items():
#     for user, ts, tweet, bertopic_label in zip(cascades['user'], cascades['ts'], cascades['word'], cascades['label']):
#         if user not in user_tweets_dict:
#             user_tweets_dict[user] = []
#         user_tweets_dict[user].append({
#             "tag": tag,
#             "bertopic_label": str(bertopic_label),
#             "user": user,
#             "ts": ts,
#             "text": tweet,
#         })

# print(len(user_tweets_dict))
# print(sum([len(tweets) for tweets in user_tweets_dict.values()]))

# 1-2. 按照用户粒度降采样

# 去掉已经采样过的用户

# down_sampling_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/input_sample.jsonl"
# down_sampling_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/generation_1/generation_1.jsonl"
down_sampling_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1/generation_1.jsonl"
regex_pattern = r"\[\d+\] ([\w\s]+)"

def read_from_topicgpt(filepath):
    user_tweets = {}
    with open(filepath, "r") as f:
        for line in f:
            info = json.loads(line)
            text = info["text"]
            mt_info = info["meta_info"]
            topic = info["responses"]
            topic = regex.compile(regex_pattern).findall(topic)
            if len(topic) > 0: topic = topic[0]
            else: topic = "None"
            if mt_info["user"] not in user_tweets: user_tweets[mt_info["user"]] = []
            user_tweets[mt_info["user"]].append({
                "user": mt_info["user"],
                "ts": mt_info["ts"],
                "text": text,
                "tag": mt_info["tag"],
                "bertopic_label": mt_info["bertopic_label"],
                "topicgpt_label": topic,
            })
    return user_tweets

user_tweets = read_from_topicgpt(down_sampling_filepath)
print(len(user_tweets))

sampled_users = list(user_tweets.keys())
print(len(sampled_users))
print(len(user_tweets_dict))

user_seeds = list(set((user_tweets_dict.keys())) - set(sampled_users))
# user_seeds = list(user_tweets_dict.keys())
print(len(user_seeds))
# NOTE: 选2000个用户进行采样
user_seeds = random.sample(user_seeds, k=2000)
print(len(user_seeds))

user_tweets_dict_sample = {}
user_tweets_dict_remain = {}
for us in user_seeds:
    unique_ = 0
    for info in user_tweets_dict[us]:
        info["unique_label"] = unique_
        unique_ += 1
    user_tweets_dict_sample[us] = random.sample(user_tweets_dict[us], k=min(len(user_tweets_dict[us]), 10))
    selected_unique_labels = []
    for info in user_tweets_dict_sample[us]:
        selected_unique_labels.append(info["unique_label"])
    user_tweets_dict_remain[us] = []
    for info in user_tweets_dict[us]:
        if info["unique_label"] not in selected_unique_labels:
            user_tweets_dict_remain[us].append(info)

print(sum([len(v) for k, v in user_tweets_dict_sample.items()]))
print(sum([len(v) for k, v in user_tweets_dict_remain.items()]))

def save_(filepath, data_dict):
    with open(filepath, 'w') as f:
        # for user, infos in user_tweets_dict.items():
        for user, infos in data_dict.items():
            for info in infos:
                result = {
                    "meta_info": {
                        "tag": info["tag"],
                        "bertopic_label": info["bertopic_label"],
                        "user": info["user"],
                        "ts": info["ts"],
                    },
                    "text": info["text"],
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

# save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_llm/input_sample.jsonl"
# remain_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_llm/input_sample_remain.jsonl"
save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/input_sample.jsonl"
remain_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/input_sample_remain.jsonl"
save_(save_filepath, user_tweets_dict_sample)
save_(remain_filepath, user_tweets_dict_remain)

# 2. 生成topicgpt的输入数据
# 执行TopicGPT script/generation_1.py脚本

# 3. 找到自适应的话题类, 并考虑用wikipedia KB知识做相似度匹配

# 合并多个话题生成结果
# 1) aggreagate generation_1 results

# glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/generation_1_*/generation_1.jsonl"
# glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1_*/generation_1.jsonl"
glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm*/generation_1/generation_1.jsonl"

jsonl_infos = []
for filepath in glob.glob(glob_dirpath):
    print(filepath)
    with open(filepath, "r") as f:
        for line in f:
            # print(json.loads(line))
            jsonl_infos.append(json.loads(line))

print(len(jsonl_infos))

# save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/generation_1/generation_1.jsonl"
# save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1/generation_1.jsonl"
save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm_final/generation_1.jsonl"
with open(save_filepath, 'a') as f:
    for info in jsonl_infos:
        f.write(json.dumps(info, ensure_ascii=False) + '\n')

# 2) aggregate generation_1.md results
# glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/generation_1_*/generation_1.md"
# glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1_*/generation_1.md"
# glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm*/generation_1_*/generation_1.md"

# # [1] Religion (Count: 3): Mentions a controversial topic related to the classification of a religious group.
# regex_pattern = r"\[(\d+)\] ([\w\s]+) \(Count: (\d+)\): ([\w\s]+)\."

# generated_topics = {}
# for filepath in glob.glob(glob_dirpath):
#     # print(filepath)
#     with open(filepath, "r") as f:
#         for line in f:
#             parts = regex.compile(regex_pattern).findall(line)
#             if len(parts) > 0: parts = parts[0]
#             else: continue
#             _, t, cnt, desc = parts[:4]
#             if t not in generated_topics: generated_topics[t] = {"count": 0}
#             # if "count" not in generated_topics[t]: generated_topics[t]["count"] = 0
#             generated_topics[t]["count"] += int(cnt)
#             generated_topics[t]["desc"] = desc

# # 126 / 6085
# print(len(generated_topics))
# print(sum([v["count"] for k,v in generated_topics.items()]))

# # save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/generation_1/generation_1.md"
# # save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1/generation_1.md"
# save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm_final/generation_1.md"
# with open(save_filepath, 'w') as f:
#     for t, info in generated_topics.items():
#         f.write(f"[1] {t} (Count: {info['count']}): {info['desc']}.\n")

# 2-2) 由于一开始986条数据的md丢失了, 要从jsonl中重新生成md
# glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm/generation_1_*/generation_1.jsonl"
glob_dirpath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1/generation_1.jsonl"

# [1] Politics: Mentions the involvement of the President of the United States and USAID in political actions related to socialism and corruption.
regex_pattern = r"\[(\d+)\] ([\w\s]+): ([\w\s]+)\."

generated_topics = {}
for filepath in glob.glob(glob_dirpath):
    # print(filepath)
    with open(filepath, "r") as f:
        for line in f:
            # print(json.loads(line))
            info = json.loads(line)
            response = info["responses"]
            parts = regex.compile(regex_pattern).findall(response) 
            if len(parts) > 0: 
                parts = parts[0]
                _, t, desc = parts[:3]
                if t not in generated_topics: generated_topics[t] = {"count": 0}
                generated_topics[t]["count"] += 1
                generated_topics[t]["desc"] = desc
            else:
                pass
                # if "None" not in generated_topics: generated_topics["None"] = {"count": 0}
                # generated_topics["None"]["count"] += 1

# 141 -> 142(+None) / 5666 -> 8403
print(len(generated_topics))
print(sum([v["count"] for k,v in generated_topics.items()]))

# save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1/generation_1.md"
# with open(save_filepath, 'w') as f:
#     for t, info in generated_topics.items():
#         f.write(f"[1] {t} (Count: {info['count']}): {info['desc']}.\n")

# 3) 使用TopicGPT自带的合并脚本试试看
# 结论: 聊胜于无

# 4) 分析话题分布情况

sorted_gt = sorted(generated_topics.items(), key=lambda x: x[1]["count"], reverse=True)
print(sum([v["count"] for k, v in sorted_gt]))

# 结论: >50的有20个, >30的有33个; 从降采样的这1w条推文上来看有效话题的数量级是可接受的
# 可以等后续另外2000个用户的2w条推文的话题分布情况再评估
for k, v in sorted_gt:
    print(k, v)

# 5-2) 用Wikipedia归纳分析TopicGPT生成的话题类型

# 5-2-1. Read from Topic Lists

# def get_subcategories(url, level):
#     # Send a GET request to the URL
#     try:
#         response = requests.get(url, proxies=proxies)
#     except Exception as e:
#         print(e)
#         return {}

#     # Parse the HTML content of the page with BeautifulSoup
#     soup = BeautifulSoup(response.text, 'html.parser')
    
#     # Find the div containing the subcategories
#     container = soup.find('div', id='mw-subcategories')
    
#     # Extract all links within this div
#     subcategory_links = container.find_all('a') if container else []
    
#     # Collect subcategory names and links
#     topics = {}
#     for link in subcategory_links:
#         t = link.get('href')
#         if t is None or level >= 2: continue
#         if "Main_topic_articles" in t: continue
        
#         sublink = "https://en.wikipedia.org" + t
#         topics[t] = get_subcategories(sublink, level + 1)

#     return topics

# # URL of the Wikipedia category page
# url = 'https://en.wikipedia.org/wiki/Category:Main_topic_classifications'
# proxies = {
#     "http": "http://127.0.0.1:1087",
#     "https": "http://127.0.0.1:1087"
# }
# subcategories = get_subcategories(url, 0)
# response = requests.get(url, proxies=proxies)
# Print the subcategories
# for name, link in subcategories.items():
#     print(f"{name}: {link}")

filepath = "/root/pyHeter-GAT/preprocess-llm/output/wikipedia_class.txt"

first_topics = []
second_topics = []
# ft_st_mapping = {}
st_ft_mapping = {}
last_ft = None
with open(filepath, 'r') as f:
    for line in f:
        if line[0] == ' ':
            # ft_st_mapping[last_ft].append(line.strip())
        
            st_ft_mapping[line.strip()] = last_ft
            second_topics.append(line.strip())
        else:
            last_ft = line.strip()
            # ft_st_mapping[last_ft] = []
            first_topics.append(line.strip())

# add first topics
for ft in first_topics:
    st_ft_mapping[ft] = ft

# print(ft_st_mapping)
# print(first_topics.keys())

# 5-2-2. Prepare sBert encoding
from sentence_transformers import SentenceTransformer, util

sbert = SentenceTransformer("/remote-home/share/dmb_nas/wangzejian/LLM_GNN/all-MiniLM-L6-v2")

f_topic_mp = {}
f_topic_embs = sbert.encode(first_topics)
for tp, emb in zip(first_topics, f_topic_embs):
    f_topic_mp[tp] = emb

s_topic_mp = {}
s_topic_embs = sbert.encode(second_topics)
for tp, emb in zip(second_topics, s_topic_embs):
    s_topic_mp[tp] = emb

# 5-2-3. Find the most related wikipedia topics for TopicGPT-generated results
gts = list(generated_topics.keys())

gt_ft_mapping = {}
gt_embs = sbert.encode(gts)
for emb, tp in zip(gt_embs, gts):
    max_sim = -1
    max_tp = ""
    for f_tp, f_emb in f_topic_mp.items():
        sim = util.pytorch_cos_sim(emb, f_emb)
        if sim > max_sim:
            max_sim = sim
            max_tp = f_tp

    for s_tp, s_emb in s_topic_mp.items():
        # if s_tp not in ft_st_mapping[max_tp]: continue
        sim = util.pytorch_cos_sim(emb, s_emb)
        if sim > max_sim:
            max_sim = sim
            max_tp = s_tp
    # gt_mapping[tp].append(max_sp)
    gt_ft_mapping[tp] = st_ft_mapping[max_tp]

# for k, v in gt_ft_mapping.items():
#     print(k, v)

save_pickle(gt_ft_mapping, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/gt_ft_mapping.data")

topic_dist = {}
for gt, ft in gt_ft_mapping.items():
    if ft not in topic_dist: topic_dist[ft] = 0
    topic_dist[ft] += generated_topics[gt]["count"]

def sort_dict(m: dict) -> dict:
    return dict(sorted(m.items(), key=lambda x: x[1], reverse=True))

tot = sum([v for k,v in topic_dist.items()])
for k,v in sort_dict(topic_dist).items():
    print(k, v, round(100*v/tot, 2))

# Total: 19(Major+8(Minor)
# First-Class: Society, Politics, Economy, Trade, Sports, Law, Nature, Education, Culture, Universe, Entertainment, Technology
# options: Military, Time, 

# additional: Humanities, Engineering, Mass_media, Business, Health, Government, Life
# Academic_disciplines, Food_and_drink, Human_behavior, Knowledge, Geography, Energy

# 4. 生成话题通道

# 回想一下之前怎么构造的图

# 参考utils/graph.py中的build_heteredge_mats函数
# 1. 按照级联粒度，使用滑动窗口来构造话题通道
# P.S. 降采样了12: news_&_social_concern话题，因为它在总推文中的占比约为43w/52w
# 2. 按照话题粒度合并不同级联内的话题通道
# 3. 将每个话题的图和原始拓扑图合并
# Option: 4. 采用Motif方式做数据增强

# 4-1. Prepare input from TopicGPT
# P.S. start from 2000 seed users (out of 10000 total users)

result_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1/generation_1.jsonl"
results = []
with open(result_filepath, 'r') as f:
    for line in f:
        results.append(json.loads(line))

# 4-1-2. Prepare topic to interest mappings
gt_ft_mapping = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/gt_ft_mapping.data")

# 4-2. generate simedges from interest-aware cascades

def generate_interest_cascades(data_dict: list, remove: bool = False):
    regex_pattern = r"\[(\d+)\] ([\w\s]+): ([\w\s]+)\."

    interest_cascades = {}
    for elem in data_dict:
        response = elem["responses"]
        parts = regex.compile(regex_pattern).findall(response)
        if len(parts) > 0: parts = parts[0]
        else: continue
        _, t, _ = parts[:3]
        interest = gt_ft_mapping[t]
        if interest not in interest_cascades: interest_cascades[interest] = []
        mt_info = elem["meta_info"]
        interest_cascades[interest].append((mt_info["user"], int(mt_info["ts"])))
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

days = 7
time_distance = 3600 * 24 * days
save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/topic_diffusion_graph_full_windowsize{td}.data"
save_filepath = save_filepath.format(td=days)
print(save_filepath)
# graph_d = convert_to_graph(
#     generate_simedges(generate_interest_cascades(results, remove=True), window_size))
# save_pickle(graph_d, save_filepath)

interest_cascades = generate_interest_cascades(results, remove=True)

simedges = generate_simedges(interest_cascades, time_distance)
# select top20 interests
len_mp = {k:len(v) for k,v in simedges.items()}
for k in list(sort_dict(len_mp).keys())[20:]:
    # print(k, len(simedges[k]))
    simedges.pop(k)

u2idx = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/u2idx.data")
# print(len(u2idx))
original_edges = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/edges.data")
# print(len(original_edges))
graph_d = convert_to_graph(simedges, user_size=len(u2idx), originial_edges=original_edges)
for key, graph in graph_d.items():
    print(graph.edge_index.size())
save_pickle(graph_d, save_filepath)
