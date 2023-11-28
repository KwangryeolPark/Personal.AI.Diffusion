import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", num_layers=3):
        """_summary_
            A block of convolutional layers with ReLU activation.
        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
            padding (int, optional): _description_. Defaults to 1.
            num_layers (int, optional): _description_. Defaults to 3.
        """
        super().__init__()
        assert num_layers >= 1, "num_layers must be greater than or equal to 1"
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)   
        ))
        for _ in range(num_layers-1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)   
            ))
    
    def forward(self, input):
        out = input
        for layer in self.conv_layers:
            out = layer(out)
        return out

class Encoder(nn.Module):
    """_summary_
        One UNet block consists of 3 convolutional layers with ReLU activation.
        The output is (out, pool(out))
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.        
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, num_layers=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, input):
        out = self.conv(input)
        return (out, self.pool(out))

class Decoder(nn.Module):
    """_summary_
        One UNet block consists of 3 convolutional layers with ReLU activation.
        The output is (out, pool(out))
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.        
    """
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=skip_channels,
                                        kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = ConvBlock(in_channels=skip_channels*2, out_channels=out_channels, num_layers=3)
    
    def forward(self, input):
        skip, x = input
        out = self.up_conv(x)
        out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        return out

class UNetBlock(nn.Module):
    """_summary_
        One UNet block consists of 3 convolutional layers with ReLU activation.
        The output is (out, pool(out))
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.        
    """
    def __init__(self, in_channels, out_channels, skip_channels=None, mode="encoder"):
        super().__init__()
        
        mode = mode.lower()
        if mode == "encoder":
            self.layer = Encoder(in_channels=in_channels, out_channels=out_channels)
        elif mode == "decoder":
            self.layer = Decoder(in_channels=in_channels, out_channels=out_channels, skip_channels=skip_channels)
    
    def forward(self, input):
        return self.layer(input)