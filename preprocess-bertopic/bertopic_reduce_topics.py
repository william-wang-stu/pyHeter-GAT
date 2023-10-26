from utils.utils import load_pickle, save_pickle, flattern
from utils.utils import logger

docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_len2263914_aggby_timeline.pkl")
docs = flattern(docs)

topic_model = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_ctfidf_uint8_rawtext_partial_subg.pkl")
topic_model.reduce_topics(docs, nr_topics='auto')
logger.info("Completed...")
