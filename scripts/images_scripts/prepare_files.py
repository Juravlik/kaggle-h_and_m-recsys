import torch
import os
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import time
from torchvision import datasets
from torchvision import transforms as pth_transforms
from scripts.images_scripts.index import FlatFaissIndex
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import pickle
from .utils import open_image_RGB


DEVICE = torch.device('cuda:0')
IMAGES_FOLDER_PATH = '/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/resized_images/'

PATH_TO_FOLDER_TO_SAVE_RANK_MODEL = '/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/compressed_dataset/index/'

files = [f.replace('.jpg', '') for f in listdir(IMAGES_FOLDER_PATH) if isfile(join(IMAGES_FOLDER_PATH, f))]

artircles = pd.read_csv('/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/initial_dataset/articles.csv',
                        dtype={'article_id': str})

article_id_int = pd.read_pickle('/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/compressed_dataset/article_id_int.pickle')

articles = artircles[artircles['article_id'].isin(files)]
# articles[['article_id']].to_parquet('/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/compressed_dataset/articles_index.parquet',
#                                     index=False)

articles_list = articles['article_id'].values.tolist()

model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(DEVICE)
feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')

index = FlatFaissIndex(dimension=384, device=DEVICE)

i = 0

article_id_embeddings = {}


for article in tqdm(articles_list):

    path_to_image = os.path.join(IMAGES_FOLDER_PATH, article) + '.jpg'

    inputs = feature_extractor(images=open_image_RGB(path_to_image),
                               return_tensors="pt").to(DEVICE)

    outputs = model(inputs['pixel_values'])
    outputs = outputs.detach().cpu().numpy()[0]
    index.add_batch(np.array([outputs]))

    article_id_embeddings[article_id_int[article]] = outputs


index.build_index()

index.save_ranking_model(PATH_TO_FOLDER_TO_SAVE_RANK_MODEL)

with open(os.path.join(PATH_TO_FOLDER_TO_SAVE_RANK_MODEL, 'embeddings.pickle'), 'wb') as handle:
    pickle.dump(article_id_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

