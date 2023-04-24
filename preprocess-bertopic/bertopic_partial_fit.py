from lib.log import logger
from utils.utils import load_pickle, save_pickle, sample_docs_foreachuser2, flattern
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from river import stream, cluster
from bertopic.vectorizers import OnlineCountVectorizer, ClassTfidfTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
# from hdbscan import HDBSCAN
# from umap import UMAP
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
import argparse
import numpy as np

docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_process_len2263914_aggby_timeline.pkl")
num_timeline = len(docs)
docs = flattern(docs)
embs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/_pretrained_embeddings_rawtexts_process_len2263914.pkl")
docs = np.array(docs); embs = np.array(embs)
logger.info(f"timeline={num_timeline}, docs={len(docs)}, embs={len(embs)}")

# Create instances of GPU-accelerated UMAP and HDBSCAN
model_args = {
    'pca.n_comp': 5,
    'mbk.n_cluster': num_timeline,
    
    'umap.n_comp': 5,
    'umap.n_neigh': 200,
    'hdbscan.min_cluster_size': 10,
    'hdbscan.min_sample': 10,

    'bertopic.min_topic_size': 300,
}
assert model_args["mbk.n_cluster"] == 40239

# umap_model = UMAP(n_components=model_args['umap.n_comp'], n_neighbors=model_args["umap.n_neigh"], min_dist=0.0)
# hdbscan_model = HDBSCAN(min_samples=model_args["hdbscan.min_sample"], cluster_selection_method='eom', gen_min_span_tree=True, prediction_data=True)
# # hdbscan_model = HDBSCAN(min_cluster_size=model_args["hdbscan.min_cluster_size"], min_samples=model_args["hdbscan.min_sample"], cluster_selection_method='eom', gen_min_span_tree=True, prediction_data=True)
# vectorizer_model = CountVectorizer(ngram_range=(1,3), stop_words='english')

# model = BERTopic(language="multilingual", nr_topics="auto", top_n_words=20, min_topic_size=model_args["bertopic.min_topic_size"],
#     umap_model=umap_model, hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer_model,
# )

umap_model = IncrementalPCA(n_components=model_args["pca.n_comp"])
hdbscan_model = MiniBatchKMeans(n_clusters=model_args["mbk.n_cluster"], random_state=0)
vectorizer_model = OnlineCountVectorizer(ngram_range=(1,3), stop_words="english", decay=0.01)
model2 = BERTopic(language="multilingual", nr_topics="auto", top_n_words=20, min_topic_size=model_args["bertopic.min_topic_size"],
    umap_model=umap_model, hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
)

suffix = f"_partial"
topic_suffix = f"subg_umap_ncomp{model_args['pca.n_comp']}_mbk_ncluster{model_args['mbk.n_cluster']}_bertopic_mintopicsize{model_args['bertopic.min_topic_size']}"
# topic_suffix = f"subg_umap_ncomp{model_args['pca.n_comp']}_hdbscan_minsample{model_args['hdbscan.min_sample']}_bertopic_mintopicsize{model_args['bertopic.min_topic_size']}"
# topic_suffix = f"subg_umap_ncomp{model_args['pca.n_comp']}_hdbscan_minclustersize{model_args['hdbscan.min_cluster_size']}_minsample{model_args['hdbscan.min_sample']}_bertopic_mintopicsize{model_args['bertopic.min_topic_size']}"
logger.info("Fitting...")

# topic_labels = []
N_FRAC = 30
for idx in range(N_FRAC):
    partial_docs = docs[int(idx*len(docs)/N_FRAC):int((idx+1)*len(docs)/N_FRAC)]
    partial_embs = embs[int(idx*len(embs)/N_FRAC):int((idx+1)*len(embs)/N_FRAC)]
    logger.info(f"idx={idx}, docs={len(partial_docs)}, embs={len(partial_embs)}")

    model2.partial_fit(partial_docs, partial_embs)
    save_pickle(model2, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model{idx}_rawtext{suffix}_{topic_suffix}.pkl")

# model.topics_ = topic_labels
save_pickle(model2, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/bertopic/topic_model_rawtext{suffix}_{topic_suffix}.pkl")
logger.info("Completed...")
