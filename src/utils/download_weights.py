'''
    Utility functions for loading weights.
'''

import gdown

def download_weights(checkpoint_fpath):
    '''
        Downloads weights from Google Drive.
    '''
    
    gdown.download('https://drive.google.com/uc?id=10ieSMMJzE9EeiPIF3CMzeT4timiQTjHV', checkpoint_fpath, quiet=False)