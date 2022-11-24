'''
    Utility functions for loading weights.
'''

import gdown

def download_weights(checkpoint_fpath, model_size='L'):
    '''
        Downloads weights from Google Drive.
    '''

    download_id = '12h-NgryAf6zZdA1KclHdfzU35D1icjEp' if model_size.strip().upper() == 'L' else '1p91KBj-oUmuMfG2Gc33tEN5Js5HpV8YH'
    
    gdown.download(f'https://drive.google.com/uc?id={download_id}', checkpoint_fpath, quiet=False)