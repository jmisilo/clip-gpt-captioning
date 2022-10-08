import os
from data.dataset import MiniFlickrDataset

if __name__ == '__main__':
    dataset = MiniFlickrDataset(os.path.join('data', 'processed', 'dataset.pkl'))