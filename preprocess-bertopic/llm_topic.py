from utils.utils import load_pickle, save_pickle, flattern, check_memusage_GB
from lib.log import logger
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from scipy.special import expit
import numpy as np
import torch

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--idx', type=int, default=0, help="")
# args = parser.parse_args()

MODEL = f"cardiffnlp/tweet-topic-21-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
if torch.cuda.is_available():
    model = model.to('cuda')

class_mapping = model.config.id2label

docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/norm_texts_rawtexts_len2263914_aggby_timeline.pkl")
docs = flattern(docs)

N_FRAC = 12000
suffix = f"_model_{MODEL.split('/')[1]}"

last, cnt = 0, 200
full_embeds, full_labels = [], []
for idx in range(N_FRAC):
    if idx >= last + cnt:
        last = idx
        logger.info(f"idx={idx}, last={last}, embeds={len(full_embeds)}, labels={len(full_labels)}, memory usage={check_memusage_GB()}")
        save_pickle(full_embeds, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm-topic/topic_distrs{idx}_llm_normtexts{suffix}.pkl")
        save_pickle(full_labels, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm-topic/topic_labels{idx}_llm_normtexts{suffix}.pkl")
    
    partial_docs = docs[int(len(docs)/N_FRAC*idx):int(len(docs)/N_FRAC*(idx+1))]
    # logger.info(f"len={len(partial_docs)}, start:end={int(len(docs)/N_FRAC*idx)}:{int(len(docs)/N_FRAC*(idx+1))}")
    
    with torch.no_grad():
        tokens = tokenizer(partial_docs, return_tensors='pt', padding=True)
        if torch.cuda.is_available():
            tokens = {
                'input_ids': tokens['input_ids'].to('cuda'),
                # 'token_type_ids': tokens['token_type_ids'].to('cuda'),
                'attention_mask': tokens['attention_mask'].to('cuda'),
            }
        output = model(**tokens)
        embeds = output.logits.detach().cpu()
        scores = expit(embeds)
        labels = np.array([np.where(score==max(score))[0][0] for score in scores])
        # logger.info(f"embeds={len(embeds)}, scores={len(scores)}, unique={np.unique(scores, return_counts=True)}")
    full_embeds.extend(embeds); full_labels.extend(labels)

    del tokens, output
    torch.cuda.empty_cache()

logger.info(f"idx={idx}, last={last}, embeds={len(full_embeds)}, labels={len(full_labels)}, memory usage={check_memusage_GB()}")
save_pickle(full_embeds, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm-topic/topic_distrs_llm_normtexts{suffix}.pkl")
save_pickle(full_labels, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm-topic/topic_labels_llm_normtexts{suffix}.pkl")
logger.info("Completed...")
