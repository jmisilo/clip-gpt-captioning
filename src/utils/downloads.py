'''
    Utility functions for loading weights.
'''

import gdown

def download_weights(checkpoint_fpath):
    '''
        Downloads weights from Google Drive.
    '''
    
    gdown.download('https://drive.google.com/uc?id=10ieSMMJzE9EeiPIF3CMzeT4timiQTjHV', checkpoint_fpath, quiet=False)

def download_dataset(destination_path):
    '''
        Downloads dataset from Google Drive.
    '''

    gdown.download('https://drive.google.com/uc?id=1E7lKanGE2Gakgy3mvyUal_B43BxU3vHr', destination_path, quiet=False)