import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model_ae import IconAutoencoder
from data_icons import get_data_loaders, IconDataset
from utils_vis import (save_recon_grid, save_random_samples, create_interpolation_gif, save_interp_strip,log_images_to_tensorboard)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loss_fn(loss_type):
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        recon = model(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.size(0)
    
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            total_loss += loss.item() * batch.size(0)
    
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default='src/collecting/data_processed/images')
    parser.add_argument('--splits_dir', default='src/collecting/splits')
    parser.add_argument('--artifacts_dir', default='artifacts')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--loss', default='l1', choices=['l1', 'mse'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_interp', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_data_loaders(
        args.images_dir, args.splits_dir, args.batch_size, args.num_workers
    )
    print(f"train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")
    
    model = IconAutoencoder(latent_dim=args.latent_dim).to(device)
    
    criterion = get_loss_fn(args.loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    tb_dir = os.path.join(args.artifacts_dir, 'tb_runs')
    writer = SummaryWriter(tb_dir)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"epoch {epoch+1}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader)).to(device)
                sample_recon = model(sample_batch)
                
                save_path = os.path.join(args.artifacts_dir, f'recon_epoch_{epoch}.png')
                save_recon_grid(sample_batch, sample_recon, save_path, n_images=16)
                
                log_images_to_tensorboard(writer, sample_batch, sample_recon, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.artifacts_dir, 'best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"early stop at epoch {epoch+1}")
                break
    
    torch.save(model.state_dict(), os.path.join(args.artifacts_dir, 'last.pt'))
    print(f"best val: {best_val_loss:.4f}")
    
    model.load_state_dict(torch.load(os.path.join(args.artifacts_dir, 'best.pt')))
    model.eval()
    
    print("generating random samples")
    random_path = os.path.join(args.artifacts_dir, 'random_samples.png')
    save_random_samples(model, device, args.latent_dim, random_path, n_samples=16)
    
    print("creating interpolation")
    test_ds = IconDataset(args.images_dir, os.path.join(args.splits_dir, 'test.txt'))
    img_a = test_ds[0]
    img_b = test_ds[1]
    
    gif_path = os.path.join(args.artifacts_dir, 'interp_A_to_B.gif')
    frames = create_interpolation_gif(model, img_a, img_b, device, gif_path, n_frames=args.num_interp, fps=10)
    
    strip_path = os.path.join(args.artifacts_dir, 'interp_strip.png')
    save_interp_strip(frames, strip_path, n_show=min(len(frames), 30))
    
    print(f"done")
    
    writer.close()


if __name__ == '__main__':
    main()
