import torch
import torch.nn as nn


def make_conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )


class BaselineCNN(nn.Module):
    def __init__(self, channels, dropout=0.3, embedding_dim=256):
        super().__init__()
        
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(make_conv_block(in_ch, out_ch))
            in_ch = out_ch
        
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x
