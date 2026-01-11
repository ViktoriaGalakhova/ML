import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import imageio
import os


def save_recon_grid(originals, reconstructions, save_path, n_images=16):
    n = min(n_images, originals.size(0))
    
    comparison = torch.cat([originals[:n], reconstructions[:n]])
    grid = vutils.make_grid(comparison, nrow=n, normalize=False, padding=2)
    
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    grid_np = (grid_np * 255).astype(np.uint8)
    
    img = Image.fromarray(grid_np)
    img.save(save_path)


def save_random_samples(model, device, latent_dim, save_path, n_samples=16):
    model.eval()
    with torch.no_grad():
        z_random = torch.randn(n_samples, latent_dim).to(device)
        samples = model.decoder(z_random)
        
        grid = vutils.make_grid(samples, nrow=4, normalize=False, padding=2)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        grid_np = (grid_np * 255).astype(np.uint8)
        
        img = Image.fromarray(grid_np)
        img.save(save_path)


def create_interpolation_gif(model, img_a, img_b, device, save_path, n_frames=30, fps=10):
    model.eval()
    with torch.no_grad():
        z_a = model.encoder(img_a.unsqueeze(0).to(device))
        z_b = model.encoder(img_b.unsqueeze(0).to(device))
        
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            z_interp = (1 - t) * z_a + t * z_b
            
            img_interp = model.decoder(z_interp)
            img_np = img_interp.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            frames.append(img_np)
        
        imageio.mimsave(save_path, frames, fps=fps)
    
    return frames


def save_interp_strip(frames, save_path, n_show=20):
    if n_show > len(frames):
        n_show = len(frames)
    
    step = max(1, len(frames) // n_show)
    selected = [frames[i * step] for i in range(n_show)]
    
    h, w = selected[0].shape[:2]
    strip = np.zeros((h, w * n_show, 3), dtype=np.uint8)
    
    for i, frame in enumerate(selected):
        strip[:, i*w:(i+1)*w, :] = frame
    
    img = Image.fromarray(strip)
    img.save(save_path)


def log_images_to_tensorboard(writer, originals, reconstructions, epoch, n_images=8):
    n = min(n_images, originals.size(0))
    comparison = torch.cat([originals[:n], reconstructions[:n]])
    grid = vutils.make_grid(comparison, nrow=n, normalize=False, padding=2)
    writer.add_image('recons/orig_vs_recon', grid, epoch)
