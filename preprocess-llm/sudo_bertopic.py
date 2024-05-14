from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score

import regex
import json

import pickle
import numpy as np
# import pandas as pd
import os
from openai import OpenAI
import httpx

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

# 1. Read From Supervised Results
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

user_infos = read_from_topicgpt(down_sampling_filepath)
print(len(user_infos))
print(sum([len(tweets) for user, tweets in user_infos.items()]))

# 1-2. Expand to Broader Interests
gt_ft_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/gt_ft_mapping.data"
gt_ft_mapping = load_pickle(gt_ft_filepath)

# 1-3. Get labeled data
data = []
target = []
for user, infos in user_infos.items():
    for info in infos:
        if info["topicgpt_label"] == "None": continue
        if info["topicgpt_label"] not in gt_ft_mapping: continue
        data.append(info["text"])
        # target.append(info["topicgpt_label"])
        target.append(gt_ft_mapping[info["topicgpt_label"]])

# Discretize targets
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# target = label_encoder.fit_transform(target)

print(len(data))
print(sum([len(tweets) for user, tweets in user_infos.items()]))

# 1-4. split into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 2. Train Semi-supervised BERTopic

# # Skip over dimensionality reduction, replace cluster model with classifier,
# # and reduce frequent words while we are at it.
# empty_dimensionality_model = BaseDimensionalityReduction()
# clf = LogisticRegression()
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# sbert = SentenceTransformer("/remote-home/share/dmb_nas/wangzejian/LLM_GNN/all-MiniLM-L6-v2")

# # Create a fully supervised BERTopic instance
# topic_model= BERTopic(
#     umap_model=empty_dimensionality_model,
#     hdbscan_model=clf,
#     ctfidf_model=ctfidf_model,
#     embedding_model=sbert,
# )
# topic_model = topic_model.fit(X_train, y=y_train)
# save_pickle(topic_model, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/bertopic_model.data")

# # Map input `y` to topics
# mappings = topic_model.topic_mapper_.get_mappings()
# reverse_mapping = {v: k for k, v in mappings.items()}
# # print(mappings)

# y_preds = []
# topics, _ = topic_model.transform(X_test)
# # predictions = label_encoder.inverse_transform([reverse_mapping[topic] for topic in topics])
# predictions = [mappings[topic] for topic in topics]
# print(predictions)
# # y_preds.append(mappings[topic[0]])

# 3. Direct SBert Cosine Similarity

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

# X. Results

# test accuracy
print(accuracy_score(y_test, predictions))


# Z. Apply to Other unsupervised data
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

user_remain_infos = {}
user_seeds = user_infos.keys()
for user, full_infos in user_tweets_dict.items():
    if user not in user_seeds:
        user_remain_infos[user] = full_infos
        continue

    infos = user_infos[user]
    remain_infos = []
    for info in full_infos:
        flag = False
        for s_info in infos:
            if info["text"] == s_info["text"] \
                and info["tag"] == s_info["tag"] \
                and info["user"] == s_info["user"] \
                and info["ts"] == s_info["ts"]:
                flag = True
                break
        if not flag:
            remain_infos.append(info)
    # print(len(infos), len(remain_infos), len(full_infos), "{:.2f}".format(len(remain_infos) / len(full_infos)))
    # try:
    #     assert len(infos) + len(remain_infos) == len(full_infos)
    # except Exception as e:
    #     print(user, len(infos), len(remain_infos), len(full_infos))
    user_remain_infos[user] = remain_infos

print(sum([len(infos) for infos in user_remain_infos.values()]))

# user_remain_infos_cp = {}

save_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/topic_llm2/generation_1.jsonl"

with open(save_filepath, 'w') as f:
    for user, infos in user_remain_infos.items():
        testing_texts = [info["text"] for info in infos]
        test_embeddings = sbert.encode(X_test[:100])
        cosine_scores = util.pytorch_cos_sim(test_embeddings, topic_t_embeddings)
        # print(cosine_scores.shape)
        predictions = np.argmax(cosine_scores, axis=1).tolist()
        predictions = [key2idx[p] for p in predictions]

        # user_remain_infos_cp[user] = []
        for info, pred in zip(infos, predictions):
            result = {
                "meta_info": {
                    "user": info["user"],
                    "ts": info["ts"],
                    "tag": info["tag"],
                    "bertopic_label": info["bertopic_label"],
                },
                "text": info["text"],
                "topicgpt_label": pred,
            }
            # info["topicgpt_label"] = pred
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            # user_remain_infos_cp[user].append(info)
