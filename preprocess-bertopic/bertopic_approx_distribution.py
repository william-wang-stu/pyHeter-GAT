from utils.utils import load_pickle, save_pickle
from utils.utils import logger
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help="")
args = parser.parse_args()

docs = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_process_remove_hyphen.pkl")
docs = [item for sublist in docs for item in sublist]

# topic_model_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext_remove_hyphen_reduce_auto.pkl"
topic_model_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext_remove_hyphen_nodup_frac0.05_reduce_auto.pkl"
topic_model = load_pickle(topic_model_filepath)

suffix = topic_model_filepath.split('/')[-1][19:]
suffix = ".".join(suffix.split('.')[:-1])
logger.info(f"suffix={suffix}")

N_FRAC, n_docs = 10, len(docs)
test_texts = docs[int(n_docs/N_FRAC*args.idx):int(n_docs/N_FRAC*(args.idx+1))]
topic_distr, _ = topic_model.approximate_distribution(test_texts, min_similarity=1e-5)
topic_distr = topic_distr.astype(np.float16) # save space and s/l time
save_pickle(topic_distr, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_distribution{args.idx}{suffix}.pkl")
logger.info("Complete Distribution...")

# labels, probs = topic_model.transform(test_texts)
# save_pickle(labels, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label{args.idx}{suffix}.pkl")
# save_pickle(probs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob{args.idx}{suffix}.pkl")

logger.info("Completed...")

if args.idx == N_FRAC-1:
    logger.info("Merge All Pieces...")
    whole_label, whole_prob, whole_distr = [], [], []

    for idx in range(N_FRAC):
        # label = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label{idx}{suffix}.pkl")
        # prob = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob{idx}{suffix}.pkl")
        distr = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_distribution{idx}{suffix}.pkl")
        # whole_prob.extend(prob); whole_label.extend(label); whole_distr.extend(distr)
        whole_distr.extend(distr)

    # save_pickle(whole_label, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label{suffix}.pkl")
    # save_pickle(whole_prob, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob{suffix}.pkl")
    save_pickle(whole_distr, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_distribution{suffix}.pkl")
