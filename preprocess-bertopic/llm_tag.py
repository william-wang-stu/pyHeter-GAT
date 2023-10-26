from lib.log import logger
from utils.utils import load_pickle, save_pickle
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sklearn.decomposition import PCA
import argparse

pretrained_model_name = 'xlm-roberta-base'
suffix = f"_model_{pretrained_model_name}"
logger.info(suffix)
# n_user = 44896
# cascades = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/build_cascades/deg_le483_timeline_aggby_tag.pkl")

# # Prepare Pretrained-LLM
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
# model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name)
# if torch.cuda.is_available():
#     model = model.to('cuda')

# # Per Hashtag
# tag_embs = []
# user2tagids = [[] for _ in range(n_user)]
# for tag_id, tag in enumerate(sorted(cascades.keys())):
#     # Calculate Emb
#     with torch.no_grad():
#         tokens = tokenizer(tag, return_tensors='pt', padding=True)
#         if torch.cuda.is_available():
#             tokens = {
#                 'input_ids': tokens['input_ids'].to('cuda'),
#                 # 'token_type_ids': tokens['token_type_ids'].to('cuda'),
#                 'attention_mask': tokens['attention_mask'].to('cuda'),
#             }
#         output = model(**tokens)
#         embeds = output.logits.detach().cpu().numpy()
#         embeds = np.mean(embeds[0], axis=0, dtype=np.float16)
#     tag_embs.append(embeds)
#     # tag2emb[tag] = embeds
#     del tokens, output

#     # Accumulate User Appearance in Multiple Tags
#     cascade = cascades[tag]
#     for user_id, _ in cascade:
#         user2tagids[user_id].append(tag_id)

# tag_embs = np.array(tag_embs)
# save_pickle(tag_embs, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/tag_embs{suffix}.pkl")
# save_pickle(user2tagids, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_user2tagids.pkl")
# logger.info("Completed...")

# # Aggregate Tag Emb foreach User
# tag_embs = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/tag_embs{suffix}.pkl")
# user2tagids = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/basic/deg_le483_user2tagids.pkl")
# logger.info("Data Loading...")

# user2emb = []
# for tag_ids in tqdm(user2tagids):
#     if len(tag_ids) == 0:
#         emb = np.zeros(tag_embs.shape[1])
#     else:
#         emb = np.mean(tag_embs[tag_ids], axis=0)
#     user2emb.append(emb)

# user2emb = np.array(user2emb)
# save_pickle(user2emb, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llmtag_embs_aggbyuser{suffix}.pkl")
# logger.info("Completed...")

# Reduce Dimension, since xlm-roberta-base output is too largeeee!
# user2emb = load_pickle(f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/tag_embs_aggbyuser{suffix}.pkl")
tag2emb = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/tag_embs_model_xlm-roberta-base.pkl")
logger.info("Data Loading...")

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=128, help="")
args = parser.parse_args()

user2emb = tag2emb.transpose((1,0))
pca = PCA(n_components=args.dim)
pca.fit(user2emb)
tweet_features = pca.components_
tweet_features = tweet_features.transpose((1,0))

save_pickle(tweet_features, f"/remote-home/share/dmb_nas/wangzejian/HeterGAT/tweet-embedding/llm/tag_embs{suffix}_pca_dim{args.dim}.pkl")
logger.info("Completed...")
