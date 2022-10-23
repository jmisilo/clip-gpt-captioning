import torch

def load_ckp(checkpoint_fpath, model, optimizer, scheduler, scaler, device='cpu'):
    checkpoint = torch.load(checkpoint_fpath, map_location=device)

    model.load_state_dict(checkpoint['model1_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch']