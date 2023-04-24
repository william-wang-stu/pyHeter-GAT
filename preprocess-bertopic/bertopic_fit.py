from lib.log import logger
from utils.utils import load_pickle, save_pickle, sample_docs_foreachuser2, flattern
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import emoji
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--frac', type=float, default=0.5, help="")
args = parser.parse_args()
logger.info(f"Args: {args}")

# import re
# docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_processedforbert_subg.pkl")
# re_docs = []
# for texts in docs:
#     re_texts = []
#     for text in texts:
#         re_texts.append(re.sub(r'(\b_\w+)-(\w+_\b)', r'\1_\2', text))
#     re_docs.append(re_texts)
# save_pickle(re_docs, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_re_subg.pkl")

# docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_processedforbert_subg.pkl")
# docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/raw_texts_aggby_user_filter_lt2words_process_remove_hyphen.pkl")
docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/norm_texts_rawtexts_len2263914_aggby_timeline.pkl")
docs = flattern(docs)
# docs = sample_docs_foreachuser2(docs, sample_frac=args.frac)
import random
docs = random.choices(docs, k=int(args.frac*len(docs)))
logger.info(f"len={len(docs)}")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=True)

suffix = f"_remove_hyphen_nodup_frac{args.frac:.2f}"
save_pickle(sentence_model, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_model_rawtext{suffix}.pkl")
save_pickle(docs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_sampledocs_rawtext{suffix}.pkl")
save_pickle(embeddings, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_embeddings_rawtext{suffix}.pkl")

# sentence_model = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_model_rawtext{suffix}.pkl")
# docs = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_sampledocs_rawtext{suffix}.pkl")
# embeddings = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_embeddings_rawtext{suffix}.pkl")

model_args = {
    'umap.n_comp': 5,
    'umap.n_neigh': 200,
    'hdbscan.min_sample': 10,
    'bertopic.min_topic_size': 300,
}

# Create instances of GPU-accelerated UMAP and HDBSCAN
umap_model = UMAP(n_components=model_args['umap.n_comp'], n_neighbors=model_args["umap.n_neigh"], min_dist=0.0)
hdbscan_model = HDBSCAN(min_samples=model_args["hdbscan.min_sample"], gen_min_span_tree=True, prediction_data=True)

# Pass the above models to be used in BERTopic
# demojize_texts = ['_'+demojize_text[1:-1]+'_' for demojize_text in emoji.get_emoji_unicode_dict(lang='en').keys()]
# stop_words = set(ENGLISH_STOP_WORDS) | set(demojize_texts)
# vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words=list(stop_words))
vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words='english')
model = BERTopic(language="multilingual", nr_topics="auto", top_n_words=20, min_topic_size=model_args["bertopic.min_topic_size"],
    embedding_model=sentence_model, vectorizer_model=vectorizer_model,
    umap_model=umap_model, hdbscan_model=hdbscan_model
)

topic_suffix = f"subg_umap_ncomp{model_args['umap.n_comp']}_nneigh{model_args['umap.n_neigh']}_hdbscan_minsample{model_args['hdbscan.min_sample']}_bertopic_mintopicsize{model_args['bertopic.min_topic_size']}"
logger.info("Fitting...")
topic_model = model.fit(docs, embeddings)
save_pickle(topic_model, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext{suffix}_{topic_suffix}.pkl")
logger.info("Completed...")

topic_model = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext{suffix}_{topic_suffix}.pkl")
logger.info(len(topic_model.topic_representations_))
train_texts = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/sentence_sampledocs_rawtext{suffix}.pkl")
topic_model.reduce_topics(train_texts, nr_topics="auto")
logger.info(len(topic_model.topic_representations_))
save_pickle(topic_model, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext{suffix}_reduce_auto.pkl")

topic_reprs = topic_model.topic_representations_
save_pickle(topic_reprs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_representation{suffix}_reduce_auto.pkl")
logger.info(topic_reprs)
