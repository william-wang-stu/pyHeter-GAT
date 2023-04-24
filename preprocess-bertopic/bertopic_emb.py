from utils.utils import load_pickle, save_pickle, sample_docs_foreachuser2, flattern
from utils.tweet_preprocess import preprocess_text_forbert
from utils.graph import find_tweet_by_cascade_info
from lib.log import logger
from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Tuple

# Find Tweet, and Build (tid) -> (text), (tid) -> (uid, timestamp)
# <tid2text_mp>, <tid2caselem_mp>

timelines = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/build_cascades/deg_le483_timeline_aggby_tag.pkl")
user_texts = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/text/User-Text.p")
g = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_subgraph.p")

tid = 0
tid2text_mp = {}
tid2caselem_mp = {}
tids_aggby_timeline_mp = {}
deg_le483_rawtexts_aggby_timeline_mp = {}

for id, (tag, cascades) in enumerate(timelines.items()):
    if id % 2000 == 0:
        logger.info(id)
    tids, texts = [], []
    for uid, timestamp in cascades:
        text = find_tweet_by_cascade_info(user_texts, g.vs["label"][uid], timestamp)
        if text is None:
            continue
        tid2text_mp[tid] = text
        tid2caselem_mp[tid] = {"uid":uid, "ts":timestamp}
        tids.append(tid); texts.append(text)
        tid += 1
    tids_aggby_timeline_mp[tag] = tids
    deg_le483_rawtexts_aggby_timeline_mp[tag] = texts

save_pickle(deg_le483_rawtexts_aggby_timeline_mp, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_aggby_timeline_mp.pkl")
save_pickle(tid2text_mp, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/text/deg_le483_tid2text_mp.pkl")
save_pickle(tid2caselem_mp, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/text/deg_le483_tid2caselem_mp.pkl")
logger.info("Completed...")

docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_aggby_timeline_mp.pkl")
docs = [elem for elem in docs.values()]
num = sum([len(elem) for elem in docs])
save_pickle(docs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_process_len{num}_aggby_timeline.pkl")

docs = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_process_len{num}_aggby_timeline.pkl")
docs = flattern(docs)
logger.info(len(docs))

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)
save_pickle(embeddings, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/_pretrained_embeddings_rawtexts_process_len{num}.pkl")
logger.info("Completed...")
