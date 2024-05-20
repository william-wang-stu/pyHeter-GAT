import pickle
import numpy as np
# import pandas as pd
import os
from openai import OpenAI
import httpx
import regex
import json
import random

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util

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

down_sampling_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_llm2/generation_1/generation_1.jsonl"

# regex_pattern = r"\[\d+\] ([\w\s]+)"
regex_pattern = r"\[\d+\] ([\u4e00-\u9fff]+)"

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

user_infos = read_from_topicgpt(down_sampling_filepath)
print(len(user_infos))
print(sum([len(tweets) for user, tweets in user_infos.items()]))

ch_to_en = {
    "科技": "Technology",
    "体育": "Sports",
    "社会问题": "Social Issues",
    "历史": "History",
    "娱乐": "Entertainment",
    "农业": "Agriculture",
    "政治": "Politics",
    "军事": "Military",
    "教育": "Education",
    "经济": "Economy",
    "环境": "Environment",
    "贸易": "Trade",
    "文化": "Culture",
    "地缘政治": "Geopolitics",
    "文学": "Literature",
    "食品": "Food",
    "天气": "Weather",
    "媒体": "Media",
    "社交媒体": "Social Media",
    "外交": "Diplomacy",
    "犯罪": "Crime",
    "安全": "Security",
    "健康": "Health",
    "交通": "Transportation",
    "政府": "Government",
    "国际关系": "International Relations",
    "法律": "Law",
    "宗教": "Religion",
    "旅游": "Tourism",
    "食品安全": "Food Safety",
    "商业": "Business",
    "社交": "Social Interaction"
}

en_to_ch = {v: k for k, v in ch_to_en.items()}

# gt_ft_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/gt_ft_mapping.data"
gt_ft_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_llm2/gt_ft_mapping.data"
gt_ft_mapping = load_pickle(gt_ft_filepath)

# Get labeled data
data = []
target = []
for user, infos in user_infos.items():
    for info in infos:
        if info["topicgpt_label"] == "None": continue
        if info["topicgpt_label"] not in ch_to_en: continue
        if ch_to_en[info["topicgpt_label"]] not in gt_ft_mapping: continue
        data.append(info["text"])
        # target.append(info["topicgpt_label"])
        target.append(gt_ft_mapping[ch_to_en[info["topicgpt_label"]]])

# Discretize targets
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# target = label_encoder.fit_transform(target)

print(len(data))
print(sum([len(tweets) for user, tweets in user_infos.items()]))

# split into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


# 2. Direct SBert Cosine Similarity

# Get embeddings
sbert = SentenceTransformer("/remote-home/share/dmb_nas/wangzejian/LLM_GNN/all-MiniLM-L6-v2")
embeddings = sbert.encode(X_train)

topic_embeddings = {}
for emb, label in zip(embeddings, y_train):
    if label not in topic_embeddings:
        topic_embeddings[label] = []
    topic_embeddings[label].append(emb)

for interest, emb in topic_embeddings.items():
    topic_embeddings[interest] = np.mean(emb, axis=0)
    # print(topic_embeddings[interest].shape)
key2idx = {i: k for i, k in enumerate(topic_embeddings.keys())}
topic_t_embeddings = np.array([emb for emb in topic_embeddings.values()])

# Get test embeddings
test_embeddings = sbert.encode(X_test[:100])
# print(topic_t_embeddings.shape)
# print(test_embeddings.shape)

# Get cosine similarity and Give Labels
cosine_scores = util.pytorch_cos_sim(test_embeddings, topic_t_embeddings)
# print(cosine_scores.shape)
predictions = np.argmax(cosine_scores, axis=1).tolist()
predictions = [key2idx[p] for p in predictions]
# print(predictions)
# print(y_test[:100])

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test[:100], predictions))


# Get other unlabeled data
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
print(sum([len(tweets) for user, tweets in user_tweets_dict.items()]))

candidate_users = list(
    set(list(user_tweets_dict.keys())) - \
        set(list(user_infos.keys())))
expanded_users = random.sample(candidate_users, k=2000)


# Get Remaining Info
remain_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_llm2/input_sample_remain.jsonl"
user_remains_info = {}
with open(remain_filepath, 'r') as f:
    for line in f:
        info = json.loads(line)
        user = info["meta_info"]["user"]
        if user not in user_remains_info:
            user_remains_info[user] = []
        user_remains_info[user].append(info)

print(sum([len(infos) for infos in user_remains_info.values()]))

# 1. Sample only 2000 / full users to expand
# for user in expanded_users:
#     user_remains_info[user] = user_tweets_dict[user]

# print(sum([len(infos) for infos in user_remains_info.values()]))

# 2. Sample At most 20 tweets for each user
for user in candidate_users:
    infos = user_tweets_dict[user]
    infos = [
        {
            "meta_info": {
                "tag": info["tag"],
                "user": info["user"],
                "ts": info["ts"],
                "bertopic_label": info["bertopic_label"],
            },
            "text": info["text"],
            # "topicgpt_label": "None",
        }
        for info in infos
    ]
    user_remains_info[user] = random.sample(infos, k=min(len(user_tweets_dict[user]), 20))

# t_len = []
# for user, infos in user_remains_info.items():
#     t_len.append(len(infos))

# def analyse_distribution(data):
#     for i in range(10):
#         t = np.percentile(data, i*10)
#         print(t)
# analyse_distribution(t_len)

# save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1.jsonl"
# filename = "ex_2000user"
filename = "ex_alluser_sample20"
save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/topic_llm2/generation_1/generation_1_{}.jsonl".format(filename)

with open(save_filepath, 'w') as f:
    # for user, infos in user_remain_infos.items():
    for user, infos in user_remains_info.items():
        testing_texts = [info["text"] for info in infos]
        test_embeddings = sbert.encode(X_test[:100])
        cosine_scores = util.pytorch_cos_sim(test_embeddings, topic_t_embeddings)
        # print(cosine_scores.shape)
        predictions = np.argmax(cosine_scores, axis=1).tolist()
        predictions = [key2idx[p] for p in predictions]

        # user_remain_infos_cp[user] = []
        for info, pred in zip(infos, predictions):
            info["topicgpt_label"] = pred
            f.write(json.dumps(info) + "\n")
            # user_remain_infos_cp[user].append(info)
