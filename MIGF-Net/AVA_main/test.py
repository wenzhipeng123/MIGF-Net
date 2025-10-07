import os
import sys
import torch
import numpy as np
import option
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset import AVADataset_text
from util import EDMLoss, AverageMeter
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Module import CrossAttention_module, NLB_module, DSIGF_module

from transformers import logging
logging.set_verbosity_error()

device = torch.device("cuda:0")

def theme_backbone():
    arch = 'resnet18'
    model_file = cfg.theme_model_path
    last_model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    last_model.load_state_dict(state_dict)
    model = nn.Sequential(*list(last_model.children())[:-2])

    return model

def visual_backbone():
    model = models.efficientnet_v2_s()
    pre_weights = torch.load(cfg.visual_model_path)
    model.load_state_dict(pre_weights)
    model = nn.Sequential(*list(model.children())[:-2])
    return model


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class My_bert(nn.Module):
    def __init__(self):
        super(My_bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_path)
        self.model = BertModel.from_pretrained(cfg.bert_model_path)
        self.input_ids_list = []
        self.attention_mask_list = []

    def forward(self, text):
        self.input_ids_list.clear()
        self.attention_mask_list.clear()
        for i, value in enumerate(text):
            encoded_input = self.tokenizer(value, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            self.input_ids_list.append(encoded_input["input_ids"])
            self.attention_mask_list.append(encoded_input["attention_mask"])
        input_ids = torch.cat(self.input_ids_list, dim=0).to(device)
        attention_mask = torch.cat(self.attention_mask_list, dim=0).to(device)
        # with torch.no_grad():
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

def print_model_parameters_in_million(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_all = sum(p.numel() for p in model.parameters())

    # Convert the number of parameters to millions
    total_params_in_million = total_params / 1e6
    total_params_all_in_million = total_params_all / 1e6

    print(f"Total number of trainable parameters: {total_params_in_million:.2f}M")
    print(f"Total number of parameters (including non-trainable): {total_params_all_in_million:.2f}M")


class MIGFNet(nn.Module):
    def __init__(self):
        super(MIGFNet, self).__init__()
        self.visual = visual_backbone()
        self.theme = theme_backbone()

        self.nlb = NLB_module.NonLocalBlock(1280 + 512)
        self.conv_enc = nn.Conv2d(in_channels=1280 + 512, out_channels=cfg.n_feature * 2, kernel_size=1)

        self.my_bert = My_bert()
        self.avg = nn.AdaptiveAvgPool2d((cfg.n_feature * 2, cfg.n_seq * cfg.n_seq))

        self.DSIGF_modu = DSIGF_module.DSIGF(cfg)
        self.cross_attention = CrossAttention_module.CrossAttention(cfg.n_feature * 2, cfg.n_seq)

        self.output_head = nn.Sequential(
            nn.PReLU(),
            nn.Linear(cfg.n_feature * 2, 10),
            nn.Softmax()
        )

    def forward(self, image, text):

        # with torch.no_grad():
        x_visual = self.visual(image)
        x_theme = self.theme(image)
        x_visual_theme = torch.cat([x_visual, x_theme], 1)

        f_visual_theme = self.nlb(x_visual_theme)
        f_visual_theme = self.conv_enc(f_visual_theme)
        f_visual_theme = torch.reshape(f_visual_theme, (f_visual_theme.size(0), f_visual_theme.size(1), cfg.n_seq * cfg.n_seq))

        f_text = self.my_bert(text)
        f_text = self.avg(f_text)

        encoder_tgi = self.DSIGF_modu(f_visual_theme, f_text)
        encoder_tgi = encoder_tgi[:, 1:257, :]
        encoder_igt = self.DSIGF_modu(f_text, f_visual_theme)
        encoder_igt = encoder_igt[:, 0:1, :]
        encoder_f = torch.cat([encoder_igt, encoder_tgi], 1)
        cross_attn = self.cross_attention(encoder_f).squeeze()
        x = self.output_head(cross_attn)
        return x


def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(device)
    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part():
    test_csv_path = os.path.join(cfg.label_path, 'test.csv')

    test_ds = AVADataset_text(test_csv_path, cfg.text_csv_path,
                         cfg.image_path, if_train=False)

    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    return test_loader

def test(model, loader, criterion):
    model.eval()
    test_losses = AverageMeter()
    torch.set_printoptions(precision=3)
    true_score = []
    pred_score = []

    for idx, (x, t, y) in enumerate(tqdm(loader)):
        # print_model_parameters_in_million(model)

        x = x.type(torch.FloatTensor).to(device)
        y = y.to(device).view(y.size(0), -1)
        y_pred = model(x, t)
        pscore, pscore_np = get_score(y_pred)
        tscore, tscore_np = get_score(y)
        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()
        loss = criterion(y_pred, y).float()
        test_losses.update(loss.item(), x.size(0))

    lcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)
    print('test, accuracy: {}, lcc_mean: {}, srcc_mean: {}, test_losses: {}'.format(acc, lcc_mean[0], srcc_mean[0], test_losses.avg))
    return test_losses.avg, acc, lcc_mean, srcc_mean


def start_test():
    dataloader_test = create_data_part()
    criterion = EDMLoss().to(device)
    model = MIGFNet().to(device)
    model.load_state_dict(torch.load(cfg.MIGF_model_path, map_location='cuda:0'))

    for e in range(cfg.n_epoch):
        # please set util.py r = 1 of EMD
        test_loss, tacc, tlcc, tsrcc = test(model=model, loader=dataloader_test, criterion=criterion)


if __name__ == "__main__":
    cfg = option.config
    start_test()


