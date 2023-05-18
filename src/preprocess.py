# Aminer Dataset
from utils.graph_aminer import *
from utils.graph import init_graph
# from utils.utils import split_cascades

user_ids = read_user_ids()
edges = get_static_subnetwork(user_ids)
_, new_edges = reindex_edges(user_ids, edges)
graph = init_graph(len(user_ids), new_edges, True, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre", True)

old2new_uid_mp, old2new_mid_mp = read_uid_mid_mp()

# diffusion = read_diffusion_nocontent()
# shorten_diffusion_nocontent = get_shorten_cascades_nocontent(diffusion, user_ids)
diffusion_withcontent = read_diffusion_withcontent(old2new_uid_mp, old2new_mid_mp)
shorten_cascades_withcontent = get_shorten_cascades_withcontent(diffusion_withcontent, user_ids)

midcontent = read_originial_content(old2new_mid_mp)
wordtable = read_wordtable()

# train_dict_keys, valid_dict_keys, test_dict_keys = split_cascades(shorten_cascades_withcontent, train_ratio=0.8, valid_ratio=0.1)

with open("/remote-home/share/dmb_nas/wangzejian/Aminer/TAN-weibo/train.data", 'rb') as file:
    train_data_dict = pickle.load(file)

train_data_dict_withcontent = select_and_merge_cascades(train_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
save_pickle(train_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/train_withcontent.pkl")

with open("/remote-home/share/dmb_nas/wangzejian/Aminer/TAN-weibo/valid.data", 'rb') as file:
    valid_data_dict = pickle.load(file)

valid_data_dict_withcontent = select_and_merge_cascades(valid_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
save_pickle(valid_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/valid_withcontent.pkl")

with open("/remote-home/share/dmb_nas/wangzejian/Aminer/TAN-weibo/test.data", 'rb') as file:
    test_data_dict = pickle.load(file)

test_data_dict_withcontent = select_and_merge_cascades(test_data_dict.keys(), shorten_cascades_withcontent, midcontent, wordtable)
save_pickle(test_data_dict_withcontent, "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Aminer-pre/test_withcontent.pkl")

# bertopic-preprocess/{llm-normtext, llm-topic, llm-tag}.py
