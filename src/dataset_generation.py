'''
    Script to generate the actual dataset for the model. It generates pickle file dataset with image name, image embedding and caption for each image in passed dataset.
'''

import os
import pickle
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

if __name__ == '__main__':
    # Set constants
    SEED = 100
    DATA_PATH = os.path.join('data')

    # Set random seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CLIP model and processor
    preprocessor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').vision_model.to(device)

    # Load dataset
    df = pd.read_csv(os.path.join(DATA_PATH, 'raw', 'results.csv'), sep='|')
    df.columns = [col.strip() for col in df.columns]

    df = df.drop(['comment_number'], axis=1)

    # get every 5 element of the df (5 captions per image) and save image name with corresponding captions
    ds = [(img_name, df[df['image_name'] == img_name]['comment'].values) for img_name, _ in df[0::5].to_numpy()]

    # Based on loaded dataset, create a list of (image name, image embedding, caption) tuples
    results = []
    loop = tqdm(ds, total=len(ds), position=0, leave=True)
    for img_name, cap in loop:
        try:
            img = Image.open(os.path.join(DATA_PATH, 'raw', 'flickr30k_images', img_name))

            with torch.no_grad():
                img_prep = preprocessor(images=img, return_tensors='pt').to(device)
                
                img_features = model(**img_prep)
                img_features = img_features.pooler_output
                img_features = img_features.squeeze()
                img_features = img_features.numpy()

            for c in cap:
                results.append((img_name, img_features, c[1:])) # because of the separator there is a space at the beginning of the caption
                
        except:
            print(f'Lack of image {img_name}')

    # save data into pickle file
    # img_name, img_features, caption
    with open(os.path.join(DATA_PATH, 'processed', 'dataset.pkl'), 'wb') as f:
        pickle.dump(results, f)