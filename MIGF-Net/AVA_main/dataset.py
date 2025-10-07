import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

class AVADataset_text(Dataset):
    def __init__(self, path_to_csv, texts_path, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.df_text = pd.read_csv(texts_path, encoding='ISO-8859-1')
        self.images_path = images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        scores_names = [f'score{i}' for i in range(2, 12)]
        y = np.array([row[k] for k in scores_names])

        p = y / y.sum()
        image_id = int(row['image_id'])

        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = image.resize((256, 256))
        img = self.transform(image)

        text = str(self.df_text[self.df_text['image_id'] == image_id]['image_content'].values)

        return img, text, p.astype('float32')

