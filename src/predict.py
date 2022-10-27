import os
import torch
import random
import argparse
import numpy as np
from utils.load_ckp import load_ckp
from utils.config import Config
from model.model import Net
from PIL import Image
import matplotlib.pyplot as plt

config = Config()
parser = argparse.ArgumentParser()

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='best.pt',
    help='Checkpoint name'
)

parser.add_argument(
    '-I',
    '--img-path',
    type=str,
    default='',
    help='Path to the image'
)

parser.add_argument(
    '-R',
    '--res-path',
    type=str,
    default='./data/result/prediction',
    help='Path to the results folder'
)

args = parser.parse_args()

# set seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.backends.cudnn.deterministic = True

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

if __name__ == '__main__':
    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)

    assert os.path.isfile(args.img_path), 'Image does not exist'
    assert os.path.isfile(ckp_path), 'Checkpoint does not exist'
    
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    
    img = Image.open(args.img_path)

    model = Net(
        ep_len=config.ep_len,
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        max_len=config.max_len,
        device=device
    )

    load_ckp(ckp_path, model, device=device)

    model.eval()

    with torch.no_grad():
        caption, _ = model(img)

    plt.imshow(img)
    plt.title(caption)
    plt.axis('off')

    plt.savefig(os.path.join(args.res_path, 'result.jpg'), bbox_inches='tight')

    plt.clf()
    plt.close()

    print('Generated Caption: "{}"'.format(caption))