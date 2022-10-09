import os
import wandb
import torch
import random
import numpy as np
import torch.optim as optim
from model.model import Net
from model.loops import train_epoch, valid_epoch
from utils.config import Config
from data.dataset import MiniFlickrDataset, get_loader
from utils.lr_warmup import LRWarmup
from torch.utils.data import random_split

if __name__ == '__main__':
    config = Config()

    # set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'

    model = Net(
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        max_len=config.max_len
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
    
    # build train model process with experiment tracking from wandb
    wandb.init(project='clipXgpt2 captioner', config=config.__dict__)
    wandb.watch(model, log='all')
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, scaler, optimizer, train_loader, epoch, device=device)
        valid_loss = valid_epoch(model, scaler, valid_loader, device=device)

        scheduler.step()

        # log loss to wandb
        wandb.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss
        })