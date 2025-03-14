import argparse

class Config(object):
    """Configuration of model"""
    batch_size = 32
    test_batch_size = 32
    embedding_size = 32
    hidden_size = 64
    num_epochs = 200
    max_length = 30

    n_time_interval = 40
    max_time = 120
    time_unit = 3600*24  # 3600 for Meme；1 for Weibo and Twitter

    l2_weight = 5e-5
    dropout = 0.8
    patience = 5
    freq = 5
    gpu_no = '4'
    model_name = 'hidan'
    # data_name = 'data/meme'
    learning_rate = 0.01
    optimizer = 'adam'
    random_seed = 1402

# def args_setting(config):
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-l", "--lr", type=float, help="learning rate")
#     parser.add_argument("-x", "--edim", type=int, help="embedding dimension")
#     parser.add_argument("-e", "--hdim", type=int, help="hidden dimension")
#     # /remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000
#     # /remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer
#     parser.add_argument("-d", "--data", default="Twitter-Huangxin", help="data name")
#     parser.add_argument("-g", "--gpu", help="gpu id")
#     parser.add_argument("-b", "--bs", type=int, help="batch size")
#     parser.add_argument("-t", "--tu", type=float, help="time unit")
#     args = parser.parse_args()
#     if args.lr:
#         config.learning_rate = args.lr
#     if args.edim:
#         config.embedding_size = args.edim
#     if args.hdim:
#         config.hidden_size = args.hdim
#     if args.bs:
#         config.batch_size = args.bs
#     if args.data:
#         if args.data == "Twitter-Huangxin":
#             config.data_name = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Twitter-Huangxin/sub10000/"
#         elif args.data == "Weibo-Aminer":
#             config.data_name = "/remote-home/share/dmb_nas/wangzejian/HeterGAT/Weibo-Aminer/"
#     if args.gpu:
#         config.gpu_no = args.gpu
#     if args.tu:
#         config.time_unit = args.tu
#     return config
