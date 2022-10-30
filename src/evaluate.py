'''
    Script to evaluate the model on the whole test set and save the results in folder.
'''

import argparse
import os
import random

import numpy as np

import torch
from torch.utils.data import random_split

from data.dataset import MiniFlickrDataset
from model.loops import evaluate_dataset
from model.model import Net
from utils.config import Config
from utils.load_ckp import download_weights

config = Config()
parser = argparse.ArgumentParser()

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='',
    help='Checkpoint name'
)

parser.add_argument(
    '-I',
    '--img-path',
    type=str,
    default='',
    help='Path to the test image folder'
)

parser.add_argument(
    '-R',
    '--res-path',
    type=str,
    default='',
    help='Path to the results folder'
)

parser.add_argument(
    '-T', 
    '--temperature',
    type=float,
    default=1.0,
    help='Temperature for sampling'
)

args = parser.parse_args()

ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

assert os.path.exists(args.img_path), 'Path to the test image folder does not exist'

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

if __name__ == '__main__':
    model = Net(
        ep_len=config.ep_len,
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        max_len=config.max_len,
        device=device
    )

    dataset = MiniFlickrDataset(os.path.join('data', 'processed', 'dataset.pkl'))

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    _, _, test_dataset = random_split(dataset, [config.train_size, config.val_size, config.test_size])

    if not os.path.exists(config.weights_dir):
        os.makedirs(config.weights_dir)

    if not os.path.isfile(ckp_path):
        download_weights(ckp_path)

    checkpoint = torch.load(ckp_path, map_location=device)
    model.load_state_dict(checkpoint)    

    save_path = os.path.join(args.res_path, args.checkpoint_name[:-3])

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    evaluate_dataset(model, test_dataset, args.img_path, save_path, args.temperature)