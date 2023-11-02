
import torch
from torch import nn
from  torch.nn import  functional as F
from  torch import  optim

import torchvision
from matplotlib import pyplot as plt
from  utils import plot_curve,one_hot,plot_image
batch_size = 512
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=torchvision.transforms.Compose(
    [ torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3801,))]
    )),batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=torchvision.transforms.Compose(
    [ torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,),(0.3801,))]
    )),batch_size=batch_size,shuffle=True)

x,y=next(iter(train_loader))

print(x.shape.y.shape,x.min(),y.min())

plot_image(x,y,'image sample')