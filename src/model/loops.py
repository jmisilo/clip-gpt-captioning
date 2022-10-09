import torch
from tqdm import tqdm

def train_epoch(model, scaler, optimizer, loader, epoch, device='cpu'):
    model.train()

    total_loss = 0

    loop = tqdm(loader, total=len(loader))
    loop.set_description(f'Epoch: {epoch} | Loss: ---')
    for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

        img_emb, cap, att_mask = img_emb.to(device), cap.to(device), att_mask.to(device)

        with torch.cuda.amp.autocast():
            loss = model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        total_loss += loss.item()

        loop.set_description(f'Epoch: {epoch} | Loss: {total_loss / (batch_idx + 1):.3f}')
        loop.refresh()

    return {
        'loss': total_loss / (batch_idx + 1)
    }

def valid_epoch(model, loader, device='cpu'):
    model.eval()

    total_loss = 0

    loop = tqdm(loader, total=len(loader))
    loop.set_description(f'Validation Loss: ---')
    for batch_idx, (img_emb, cap, att_mask) in enumerate(loop):

        img_emb, cap, att_mask = img_emb.to(device), cap.to(device), att_mask.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():

                loss = model.train_forward(img_emb=img_emb, trg_cap=cap, att_mask=att_mask)

                total_loss += loss.item()
                
                loop.set_description(f'Validation Loss: {total_loss / (batch_idx + 1):.3f}')
                loop.refresh()

    return {
        'loss': total_loss / (batch_idx + 1)
    }