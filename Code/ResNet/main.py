# model script for ResNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyResNet(nn.Module):
    # General 
    def __init__(self, in_channels, out_channels):
        super(MyResNet, self).__init__()

        