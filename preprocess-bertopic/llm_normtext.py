from utils.utils import load_pickle, save_pickle
from lib.log import logger
from utils.TweetNormalizer import normalizeTweet
import torch.nn.functional as F
import torch
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

docs = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_rawtexts_len2263914_aggby_timeline.pkl")
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

norm_docs = []
token_ids = []
for idx, texts in enumerate(docs):
    if idx and idx % 1000 == 0:
        logger.info(idx)
    norm_texts = []
    batch_ids = []
    for text in texts:
        norm_text = normalizeTweet(text)
        norm_texts.append(norm_text)

        input_ids = torch.tensor([tokenizer.encode(norm_text)])
        batch_ids.append(input_ids)
    norm_docs.append(norm_texts)

    max_len = max([len(elem[0]) for elem in batch_ids])
    batch_ids = [F.pad(elem, pad=(0,max_len-len(elem[0]))) for elem in batch_ids]
    batch_ids = torch.cat(batch_ids, dim=0)
    token_ids.append(batch_ids)

max_len = max([elem.shape[1] for elem in batch_ids])
batch_ids = [F.pad(elem, pad=(0,max_len-elem.shape[1])) for elem in batch_ids]
batch_ids = torch.cat(batch_ids, dim=0)
logger.info(batch_ids.shape)

save_pickle(norm_docs, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/norm_texts_rawtexts_len2263914_aggby_timeline.pkl")
save_pickle(batch_ids, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/batch_text_ids_llm_normtexts.pkl")
logger.info("Completed...")
