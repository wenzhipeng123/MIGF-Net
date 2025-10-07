import json

""" configuration json """
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

# config file
config = Config({
    # device
    'num_workers': 5,
    'prefetch_factor': 2,

    # data
    'image_path': 'D:/paper code/AADB/AADB_datasetImages_originalSize',  # directory to images (Set your path)
    'text_path': 'E:/dataset/AADB/AADB_datasetImages_originalSize_text',  # directory to texts (Set your path)
    'label_path': './dataset',  # directory to csv_folder
    'batch_size': 8,

    'seed': 1,

    # MIGF structure
    'n_feature': 128,
    'n_seq': 8,
    'theme_model_path': '../Checkpoint/Module_Checkpoint/Theme_Model/resnet18_places365.pth.tar', # (Set your path)
    'visual_model_path': 'C:/Users/啊鹏/.cache/torch/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth', # (Set your path)
    'bert_model_path': 'D:/paper code/huggingface bert-base-cased', # (Set your path)
    'MIGF_model_path': '../Checkpoint/AADB/new-epoch_1_SignificanceResult(statistic=0.8421)_PearsonRResult(statistic=0.8554)_val_loss_0.0360_.pt', # (Set your path)

    'n_enc_seq': 8 * 8,  # input feature map dimension (N = H*W) from backbone
    'n_layer': 5,  # number of encoder layers
    'd_hidn': 256,  # input channel of encoder (input: C x N)
    'i_pad': 0,
    'd_ff': 384,  # feed forward hidden layer dimension
    'd_MLP_head': 1152,  # hidden layer of final MLP
    'n_head': 6,  # number of head (in multi-head attention)
    'd_head': 384,  # channel of each head -> same as d_hidn
    'dropout': 0.1,  # dropout ratio
    'emb_dropout': 0.1,  # dropout ratio of input embedding
    'layer_norm_epsilon': 1e-12,
    'n_output': 1,  # dimension of output
    'Grid': 10,  # grid of 2D spatial embedding

    # optimization & training parameters
    'n_epoch': 300,  # total training epochs
    'learning_rate': 0.00002,  # initial learning rate
    'adam_betas': (0.9, 0.999),
    'lr_step_size': 5,
    'lr_gamma': (0.9, 0.999),
    'save_freq': 1,  # save checkpoint frequency (epoch)
    'val_freq': 5,  # validation frequency (epoch)

    # load & save checkpoint
    'save_path': '../Checkpoint/AADB',  # directory for saving checkpoint (Set your path)
})