import torch
import torchvision
import torchvision.transforms as transforms
from ShuffleNet import ShuffleNet
import torch.nn as nn
import torch.nn.functional as F
import time
from torchstat import stat
import torchvision.models as models



model = ShuffleNet(groups=8, in_channels=3, num_classes=10, scale_factor=0.5)
stat(model, (3, 32, 32))

