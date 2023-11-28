import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # add parent dir to sys.path

import torch
import torch.nn as nn
from modules.unet_block import UNetBlock

class EmbeddingUNet(nn.Module):
    def __init__(self, input_size):
        assert input_size == 32 
        super().__init__()
        self.input_size = input_size
        
        self.embedding = nn.Embedding(2048, input_size*input_size*3)
        
        self.encoder1 = UNetBlock(in_channels=6, out_channels=64)
        self.encoder2 = UNetBlock(in_channels=64, out_channels=128)
        self.encoder3 = UNetBlock(in_channels=128, out_channels=256)
        
        self.decoder1 = UNetBlock(in_channels=256, out_channels=512, skip_channels=256, mode='decoder')
        self.decoder2 = UNetBlock(in_channels=512, out_channels=256, skip_channels=128, mode='decoder')
        self.decoder3 = UNetBlock(in_channels=256, out_channels=128, skip_channels=64, mode='decoder')
        
        self.out_layer = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
    
    def forward(self, input, timestamp):
        batch_size = input.shape[0]
        out = self.embedding(timestamp)
        out = out.view(-1, 3, self.input_size, self.input_size)
        out = out.repeat(batch_size, 1, 1, 1)
        out = torch.cat((out, input), dim=1)
                
        out1 = self.encoder1(out)
        out2 = self.encoder2(out1[1])
        out3 = self.encoder3(out2[1])
        
        out4 = self.decoder1(out3)
        out5 = self.decoder2((out2[0], out4))
        out6 = self.decoder3((out1[0], out5))
        
        out7 = self.out_layer(out6)
        
        return out7
