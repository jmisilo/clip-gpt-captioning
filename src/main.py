import os
import wandb
import torch
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model.model import Net
from model.train_epoch import train_epoch
from utils.config import Config
from data.dataset import MiniFlickrDataset, get_loader
from utils.lr_warmup import LRWarmup

if __name__ == '__main__':
    # set seed
    SEED = 100

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    config = Config()
    dataset = MiniFlickrDataset(os.path.join('data', 'processed', 'dataset.pkl'))

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'

    model = Net(
        num_layers=config.num_layers, 
        n_heads=config.n_heads, 
        forward_expansion=config.forward_expansion, 
        dropout=config.dropout, 
        max_len=config.max_len
    )

    loader = get_loader(
        dataset, 
        bs_exp=config.batch_size_exp, 
        shuffle=True, 
        num_workers=config.num_workers if is_cuda else 0,
        pin_memory=is_cuda
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
    scaler = torch.cuda.amp.GradScaler()
    
    # build train model process with experiment tracking from wandb
    # wandb.init(project='clipXgpt2 captioner', config=config)
    for epoch in range(config.epochs):

        train_loss = train_epoch(model, scaler, optimizer, loader, epoch, device=device)

        scheduler.step()
        
        if epoch: break

        # log loss to wandb

        # wandb.log({'loss': total_loss / len(loader)})