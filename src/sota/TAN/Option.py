class Option(object):
    def __init__(self):
        self.epoch = 100
        self.batch_size = 32
        self.d_model = 160
        self.d_inner_hid = 160
        self.n_warmup_steps = 500
        self.dropout = 0.1
        self.max_len = 501
        self.num_heads = 5
        self.doc_size = 768
        self.Temperature = 3
        self.doc = False
        self.sub_dim = self.d_inner_hid//self.num_heads 
        self.num_blocks = 3
        self.d_user_vec = self.d_model
        self.log = '/root/TAN/TAN/log/tan'
        self.save_model = '/root/TAN/TAN/model/tan'
        self.save_mode = 'best'
        self.device = 'cuda:3'
        self.user_size = 0
        self.time_unit = 64
        self.tupe = True
        self.decay = True
        self.relative = None
        self.doc = False
