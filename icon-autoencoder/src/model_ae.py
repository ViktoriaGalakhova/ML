import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, latent_dim)
        )
    
    def forward(self, x):
        h = self.conv_blocks(x)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 3 * 3),
            nn.ReLU(inplace=True)
        )
        
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        
        self.upsample = nn.Upsample(size=(50, 50), mode='bilinear', align_corners=False)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 128, 3, 3)
        h = self.deconv_blocks(h)
        h = self.upsample(h)
        return torch.sigmoid(h)


class IconAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
