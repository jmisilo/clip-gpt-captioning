'''
    Script that contains whole training process.
'''

import argparse
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler

import wandb
from data import MiniFlickrDataset, get_loader
from model import ddp_cleanup, ddp_setup, Net, Trainer
from utils import Config, LRWarmup

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

def main(rank, world_size, config, ckp_name=''):
    # more than 1 GPU
    is_cuda = torch.cuda.is_available()
    MULTIGPU = world_size > 1

    if MULTIGPU:
        ddp_setup(rank, world_size)
        device = rank
   
    else:
        device = torch.device('cuda' if is_cuda else 'cpu')

    model = Net(
        ep_len=config.ep_len,
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        max_len=config.max_len,
        device=device
    )

    if MULTIGPU:
        model = DDP(model, device_ids=[device])

    dataset = MiniFlickrDataset(os.path.join('data', 'processed', 'dataset.pkl'))

    config.train_size = int(config.train_size * len(dataset))
    config.val_size = int(config.val_size * len(dataset))
    config.test_size = len(dataset) - config.train_size - config.val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [config.train_size, config.val_size, config.test_size])

    train_loader = get_loader(
        train_dataset, 
        bs_exp=config.batch_size_exp, 
        shuffle=not MULTIGPU, 
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        sampler=DistributedSampler(train_dataset) if MULTIGPU else None
    )

    valid_loader = get_loader(
        val_dataset, 
        bs_exp=config.batch_size_exp, 
        shuffle=False, 
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda,
        sampler=DistributedSampler(val_dataset) if MULTIGPU else None
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()    

    ckp_path = os.path.join(config.weights_dir, ckp_name)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_dataset=test_dataset,
        test_path=os.path.join('data', 'raw', 'flickr30k_images'),
        ckp_path=ckp_path,
        device=device
    )

    # build train model process with experiment tracking from wandb
    wandb.init(project='clipXgpt2 captioner', config=config.__dict__)
    wandb.watch(trainer.model, log='all')
    for epoch in range(trainer.epoch, config.epochs):
        if MULTIGPU:
            trainer.set_samplers_epoch(epoch)

        trainer.train_epoch()
        trainer.valid_epoch()
        trainer.test_result()

        metadata = trainer.get_training_data()

        # log loss to wandb
        wandb.log({
            'train_loss': metadata['train_loss'],
            'valid_loss': metadata['valid_loss'],
            'lr': metadata['lr'],
            'examples': wandb.Image(metadata['examples'])
        })

        if not os.path.exists(config.weights_dir):
            os.makedirs(config.weights_dir)

        if (epoch + 1) % 10 == 0 and rank == 0:
            trainer.save_ckp(os.path.join(config.weights_dir, f'epoch_{epoch + 1}.pt'))

    ddp_cleanup()


if __name__ == '__main__':
    # check if there is no GPU - use CPU -> world_size = 1
    
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    mp.spawn(main, args=(world_size, config, ''), nprocs=world_size)