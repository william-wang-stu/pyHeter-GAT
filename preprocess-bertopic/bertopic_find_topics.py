from utils.utils import load_pickle, save_pickle, sample_docs_foreachuser2
from lib.log import logger

def find_most_similar_topics(tags):
    full_ret = {}
    for tag in tags:
        ret = topic_model.find_topics(tag, top_n=20)
        full_ret[tag] = ret
    return full_ret

timelines = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/build_cascades/deg_le483_timeline_aggby_url_tag.pkl")
tags = list(timelines.keys())
topic_model = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext_remove_hyphen_reduce_auto.pkl")

full_ret = find_most_similar_topics(tags)
save_pickle(full_ret, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/timeline_name-to-topic_model_cluster_name-timeline_url_tag-model_remove_hyphen_reduce_auto.pkl")
logger.info("Completed...")
