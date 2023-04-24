from utils.utils import load_pickle, save_pickle, flattern
from utils.utils import logger
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help="")
args = parser.parse_args()

# docs = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_process_remove_hyphen.pkl")
docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_len2263914_aggby_timeline.pkl")
docs = flattern(docs)
embs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/_pretrained_embeddings_rawtexts_process_len2263914.pkl")

# topic_model_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext_remove_hyphen_reduce_auto.pkl"
# topic_model_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext_remove_hyphen_nodup_frac0.05_reduce_auto.pkl"
topic_model_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext_partial_subg_umap_ncomp5_mbk_ncluster40239_bertopic_mintopicsize300.pkl"
topic_model = load_pickle(topic_model_filepath)

# suffix = topic_model_filepath.split('/')[-1][19:]
# suffix = ".".join(suffix.split('.')[:-1])
suffix = "_partial_fit"
logger.info(f"suffix={suffix}")

N_FRAC, n_docs = 30, len(docs)
test_texts = docs[int(n_docs/N_FRAC*args.idx):int(n_docs/N_FRAC*(args.idx+1))]
topic_distr, _ = topic_model.approximate_distribution(test_texts, min_similarity=1e-5)
topic_distr = topic_distr.astype(np.float16) # save space and s/l time
save_pickle(topic_distr, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_distribution{args.idx}{suffix}.pkl")
logger.info("Complete Distribution...")

assert n_docs == len(embs)
test_embs = embs[int(n_docs/N_FRAC*args.idx):int(n_docs/N_FRAC*(args.idx+1))]
labels, probs = topic_model.transform(test_texts, embeddings=test_embs)
save_pickle(labels, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label{args.idx}{suffix}.pkl")
save_pickle(probs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob{args.idx}{suffix}.pkl")

logger.info("Completed...")

if args.idx == N_FRAC-1:
    logger.info("Merge All Pieces...")
    whole_label, whole_prob, whole_distr = [], [], []

    for idx in range(N_FRAC):
        label = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label{idx}{suffix}.pkl")
        prob = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob{idx}{suffix}.pkl")
        distr = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_distribution{idx}{suffix}.pkl")
        if prob is not None:
            whole_prob.extend(prob)
        whole_label.extend(label); whole_distr.extend(distr)
        # whole_distr.extend(distr)

    save_pickle(whole_label, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_label{suffix}.pkl")
    save_pickle(whole_prob, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_prob{suffix}.pkl")
    save_pickle(whole_distr, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_distribution{suffix}.pkl")
    logger.info("Completed...")
