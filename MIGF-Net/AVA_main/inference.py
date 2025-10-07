import torch
import numpy as np
import option
from torch import nn
from torchvision.datasets.folder import default_loader
from torchvision import models
from torchvision import transforms
from transformers import BertTokenizer, BertModel
from AVA.module import CrossAttention_module, NLB_module, DSIGF_module
import matplotlib.pyplot as plt

from transformers import logging
logging.set_verbosity_error()

Num_Feature = 128
device = torch.device("cuda:0")
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

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

def single_inference():
    model = MIGFNet().to(device)
    # state_dict = torch.load(f'D:\paper code\TANet\TANet-main\TANet-main\code\TANet-Demo-AVA_main\pth\epoch_a2.pt')
    # print("Loaded state_dict keys:", state_dict.keys())
    # print("Model state_dict keys:", model.state_dict().keys())
    model.load_state_dict(torch.load(
        f'/AVA_main/checkpoint/new-epoch_1_SignificanceResult(statistic=0.8421)_PearsonRResult(statistic=0.8554)_val_loss_0.0360_.pt'), strict=True)
    image = default_loader(f'D:/paper code/TANet/TANet-main/TANet-main/code/TANet-Demo-AVA/data/ava_samples/image/486506.jpg')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize])
    x_tensor = transform(image)
    tensor_with_batch = x_tensor.unsqueeze(0).to(device)
    with open('D:/paper code/TANet/TANet-main/TANet-main/code/TANet-Demo-AVA/data/ava_samples/text/486506.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    y_pred = model(tensor_with_batch, content)
    score, score_np = get_score(y_pred)

    # 展示图片
    plt.imshow(image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    print(y_pred)
    print(score)


if __name__ == "__main__":
    cfg = option.config
    single_inference()


