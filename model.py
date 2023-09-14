from torch import nn
from tcn import TemporalConvNet,ConvAutoencoder
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.autoencoder = ConvAutoencoder(input_size=num_channels[-1], output_size=output_size)
        self.linear = nn.Linear(1024,128*128)
        self.sig = nn.ReLU()

    def forward(self, x):
        output = self.tcn(x)
        output = self.autoencoder(output)
        output = self.linear(output).double()
        return output
