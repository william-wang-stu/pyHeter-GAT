# Aminer Dataset
# from utils.graph_aminer import *
# from utils.graph import init_graph
from utils.utils import split_cascades, load_pickle, save_pickle
from utils.graph_aminer import *
from utils.tweet_embedding import agg_tagemb_by_user
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from scipy.special import expit
from tqdm import tqdm
import numpy as np
import torch

# user_ids = read_user_ids()
# user_ids = user_ids.values()
# edges = get_static_subnetwork(user_ids)
# _, new_edges = reindex_edges(user_ids, edges)
# graph = init_graph(len(user_ids), new_edges, True, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre", True)

# old2new_uid_mp, old2new_mid_mp = read_uid_mid_mp()

# # diffusion = read_diffusion_nocontent()
# # shorten_diffusion_nocontent = get_shorten_cascades_nocontent(diffusion, user_ids)
# diffusion_withcontent = read_diffusion_withcontent(old2new_uid_mp, old2new_mid_mp)
# shorten_cascades_withcontent = get_shorten_cascades_withcontent(diffusion_withcontent, user_ids)

# midcontent = read_originial_content(old2new_mid_mp)
# wordtable = read_wordtable()

# # train_dict_keys, valid_dict_keys, test_dict_keys = split_cascades(shorten_cascades_withcontent, train_ratio=0.8, valid_ratio=0.1)

# with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train.data", 'rb') as file:
#     train_data_dict = pickle.load(file)

# train_data_dict_withcontent = select_and_merge_cascades(train_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
# save_pickle(train_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent.pkl")

# with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid.data", 'rb') as file:
#     valid_data_dict = pickle.load(file)

# valid_data_dict_withcontent = select_and_merge_cascades(valid_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
# save_pickle(valid_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid_withcontent.pkl")

# with open("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test.data", 'rb') as file:
#     test_data_dict = pickle.load(file)

# test_data_dict_withcontent = select_and_merge_cascades(test_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
# save_pickle(test_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test_withcontent.pkl")

# NOTE: bertopic-preprocess/{llm-normtext, llm-topic, llm-tag}.py

# train_data_dict_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent.pkl"
# valid_data_dict_filepath = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid_withcontent.pkl"
# test_data_dict_filepath  = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test_withcontent.pkl"

# MODEL = f"cardiffnlp/tweet-topic-21-multi"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# class_mapping = model.config.id2label
# if torch.cuda.is_available():
#     model = model.to('cuda')

def get_topic_label(data_dict, model, tokenizer):
    for tag, cascades in tqdm(data_dict.items()):
        words = cascades['word']
        with torch.no_grad():
            tokens = tokenizer(words, return_tensors='pt', padding=True)
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
        cascades['label'] = labels

        del tokens, output
        torch.cuda.empty_cache()
    return data_dict

# train_data_dict = load_pickle(train_data_dict_filepath)
# train_data_dict_e = get_topic_label(train_data_dict)
# save_pickle(train_data_dict_e, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/train_withcontent_withlabel.pkl")

# valid_data_dict = load_pickle(valid_data_dict_filepath)
# valid_data_dict_e = get_topic_label(valid_data_dict)
# save_pickle(valid_data_dict_e, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/valid_withcontent_withlabel.pkl")

# test_data_dict = load_pickle(test_data_dict_filepath)
# test_data_dict_e = get_topic_label(test_data_dict)
# save_pickle(test_data_dict_e, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/test_withcontent_withlabel.pkl")

# data_dict = {**train_data_dict, **valid_data_dict, **test_data_dict}

# user_set = read_user_ids(train_data_dict_filepath, valid_data_dict_filepath, test_data_dict_filepath)
# midwithcontent = load_pickle(os.path.join(DATA_ROOTPATH, "Weibo-Aminer/Aminer-pre/diffusion_original_content.pkl"))
# wordtable = load_pickle("/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/Aminer-pre/wordtable.pkl")

# pretrained_model_name = 'xlm-roberta-base'
# user2emb = agg_tagemb_by_user(n_user=len(user_set), cascades=data_dict, pretrained_model_name='xlm-roberta-base')
# save_pickle(user2emb, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/llm/tag_embs_aggbyuser_model_xlm-roberta-base_pca_dim128.pkl")
