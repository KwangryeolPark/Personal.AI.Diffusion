import math
import torch
from torch import nn

class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embedding_dim=32,
        upsample=False,
    ):
        super().__init__()
        self.time_embedding = nn.Linear(time_embedding_dim, out_channels)
        
        if upsample:
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        t = self.time_embedding(t)
        t = self.relu(t)
        t = t.view(t.shape[0], t.shape[1], 1, 1)

        x = x + t
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        
        return self.transform(x)
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, dtype=torch.float) * -embeddings).to(t.device)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros(t.shape[0], t.shape[1], 1)], dim=-1)
            
        return embeddings
    
class EmbeddingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        out_channels = 3
        time_embedding_dim = 32
        
        down_channels = [64, 128, 256, 512]
        up_channels = [512, 256, 128, 64]
        
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
        )
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        self.downsample = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_embedding_dim)
            for i in range(len(down_channels)-1)
        ])
        self.upsample = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_embedding_dim, upsample=True)
            for i in range(len(up_channels)-1)
        ])
        self.out = nn.Conv2d(up_channels[-1], out_channels, 3, padding=1)
    
    def forward(self, x, t):
        t = self.time_embedding(t)
        x = self.conv0(x)
        
        downsampled = []
        for i, down in enumerate(self.downsample):
            x = down(x, t)
            downsampled.append(x)
        
        for i, up in enumerate(self.upsample):
            residual = downsampled.pop()
            x = torch.cat([x, residual], dim=1)
            x = up(x, t)
        return self.out(x)