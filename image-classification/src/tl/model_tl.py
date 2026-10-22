import torch
import torch.nn as nn
from torchvision import models


class ResNetTransfer(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.3, embedding_dim=256):
        super().__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    def get_embedding(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x
