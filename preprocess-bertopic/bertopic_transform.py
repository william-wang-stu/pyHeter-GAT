from utils.utils import load_pickle, save_pickle
from utils.utils import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help="")
args = parser.parse_args()

docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_processedforbert_subg.pkl")
docs = [item for sublist in docs for item in sublist]

frac = 10
start,end = int(len(docs)/frac*args.idx), int(len(docs)/frac*(args.idx+1))

topic_model = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_reduce_auto_merge_lt01_remove_demojize_stopwords_subg.pkl")

topics, probs = topic_model.transform(docs[start:end])
save_pickle(topics, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_labels{args.idx}_reduce_auto_merge_lt01_remove_demojize_stopwords_subg.pkl")
save_pickle(probs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_probs{args.idx}_reduce_auto_merge_lt01_remove_demojize_stopwords_subg.pkl")

logger.info("Completed...")

