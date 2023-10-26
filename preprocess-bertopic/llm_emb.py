from utils.utils import load_pickle, save_pickle
from lib.log import logger

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='base', help="")
args = parser.parse_args()

if args.model == 'base':
    MODEL = 'vinai/bertweet-base'
# NOTE: bertweet-large yields 
# RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling `cublasCreate(handle)`
# NO IDEA HOW TO HANDLE...
# elif args.model == 'large':
#     MODEL = 'vinai/bertweet-large'
elif args.model == 'covid19':
    MODEL = 'vinai/bertweet-covid19-base-uncased'
logger.info(f"MODEL={MODEL}")

bertweet = AutoModel.from_pretrained(MODEL)
if torch.cuda.is_available():
    bertweet = bertweet.to('cuda')

batch_ids = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/batch_text_ids_llm_normtexts.pkl")
if torch.cuda.is_available():
    batch_ids = batch_ids.to('cuda')

full_doc_embs = []
N_FRAC = 10000
suffix = f"_model_{MODEL[6:]}"
logger.info(suffix)

for idx in range(N_FRAC):
    if idx and idx % 1000 == 0:
        logger.info(f"idx={idx}, len={len(full_doc_embs)}")
        save_pickle(full_doc_embs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/doc_embs{idx}_llm_normtexts{suffix}.pkl")
    
    partial_batch_ids = batch_ids[int(len(batch_ids)/N_FRAC*idx):int(len(batch_ids)/N_FRAC*(idx+1))]
    # logger.info(f"len={len(partial_batch_ids)}, start:end={int(len(batch_ids)/N_FRAC*idx)}:{int(len(batch_ids)/N_FRAC*(idx+1))}")

    with torch.no_grad():
        features = bertweet(partial_batch_ids)  # Models outputs are now tuples
        features = features.last_hidden_state.cpu().numpy()
        # logger.info(features.shape)

        avg_features = np.mean(features,axis=1)
        # logger.info(avg_features.shape)
    
    full_doc_embs.extend(avg_features)

    del features, avg_features
    torch.cuda.empty_cache()

full_doc_embs = np.array(full_doc_embs)
logger.info(full_doc_embs.shape)
save_pickle(full_doc_embs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/doc_embs_llm_normtexts{suffix}.pkl")
logger.info("Completed...")
