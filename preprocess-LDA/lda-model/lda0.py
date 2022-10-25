import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils import load_pickle, save_pickle
from lib.log import logger

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import argparse

logger.info("Start")

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0, help="")
parser.add_argument('--k', type=int, default=20, help="")
args = parser.parse_args()

texts_idx = args.idx
nptexts = load_pickle(f"lda-texts/ptexts_u{texts_idx*2}0000.p")

cv = CountVectorizer(stop_words="english")
dtm = cv.fit_transform(nptexts)
logger.info(f"dtm shape={dtm.shape}")

LDA_model = LatentDirichletAllocation(n_components=args.k, max_iter=50, random_state=2022)
LDA_model.fit(dtm)

# Topic-Word
logger.info(len(LDA_model.components_[0]))
for i, topic in enumerate(LDA_model.components_):
    logger.info("THE TOP {} WORDS FOR TOPIC #{}".format(10, i))
    logger.info([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])

# Doc-Topic
doc_topic = LDA_model.transform(dtm)
# for n in range(doc_topic.shape[0]):
#     topic_pr = doc_topic[n].argsort()[-10:]
#     # topic_pr = doc_topic[n].argmax()
#     logger.info("doc: {} topic: {}".format(n,topic_pr))

save_pickle(cv,  f"cv/cv_0{texts_idx}_k{args.k}_maxiter50.p")
save_pickle(dtm, f"dtm/dtm_0{texts_idx}_k{args.k}_maxiter50.p")
save_pickle(doc_topic, f"doc2topic/doc_topic_0{texts_idx}_k{args.k}_maxiter50.p")
save_pickle(LDA_model, f"model/model_0{texts_idx}_k{args.k}_maxiter50.p")


import pyLDAvis.sklearn
panel = pyLDAvis.sklearn.prepare(LDA_model, dtm1, cv1, mds='tsne') # Create the panel for the visualization
pyLDAvis.save_html(panel, 'LDA-vis.html')
logger.info("Finish Saving to html...")

# import pyLDAvis.sklearn

# for idx in range(5, 11):
#     name = f"_0{idx}_k{45}_maxiter{50}"

#     model = load_pickle(f"model/model{name}.p")
#     dtm   = load_pickle(f"dtm/dtm{name}.p")
#     cv    = load_pickle(f"cv/cv{name}.p")

#     panel = pyLDAvis.sklearn.prepare(model, dtm, cv, mds='tsne') # Create the panel for the visualization
#     pyLDAvis.save_html(panel, f'LDAvis{name}.html')
#     logger.info("Finish Saving to html...")
