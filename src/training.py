'''
    Script that contains whole training process.
'''

import argparse
import os
import random

import numpy as np

import wandb
import torch
import torch.optim as optim
from torch.utils.data import random_split

from data.dataset import MiniFlickrDataset, get_loader
from model.model import Net
from model.loops import train_epoch, valid_epoch, test_step
from utils.config import Config
from utils.load_ckp import load_ckp
from utils.lr_warmup import LRWarmup

config = Config()
parser = argparse.ArgumentParser()

parser.add_argument(
    '-C', 
    '--checkpoint-name',
    type=str,
    default='',
    help='Checkpoint name'
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

    train_dataset, val_dataset, test_dataset = random_split(dataset, [config.train_size, config.val_size, config.test_size])

    train_loader = get_loader(
        train_dataset, 
        bs_exp=config.batch_size_exp, 
        shuffle=True, 
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda
    )

    valid_loader = get_loader(
        val_dataset, 
        bs_exp=config.batch_size_exp, 
        shuffle=False, 
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()
    
    ckp_path = os.path.join(config.weights_dir, args.checkpoint_name)
    start_epoch = load_ckp(ckp_path, model, optimizer, scheduler, scaler, device) if os.path.isfile(ckp_path) else 0

    # build train model process with experiment tracking from wandb
    wandb.init(project='clipXgpt2 captioner', config=config.__dict__)
    wandb.watch(model, log='all')
    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(model, scaler, optimizer, train_loader, epoch, device=device)
        valid_loss = valid_epoch(model, valid_loader, device=device)
        test_results = test_step(model, test_dataset, os.path.join('data', 'raw', 'flickr30k_images'))

        scheduler.step()

        # log loss to wandb
        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'lr': scheduler.get_last_lr()[0],
            'examples': wandb.Image(test_results)
        })

        if not os.path.exists(config.weights_dir):
            os.makedirs(config.weights_dir)

        if (epoch + 1) % 10 == 0: 
            torch.save(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'model1_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, 
                os.path.join(config.weights_dir, f'epoch_{epoch}.pt')
            )