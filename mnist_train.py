import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt
from utils import plot_curve, one_hot, plot_image

batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3801,))]
    )), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3801,))]
    )), batch_size=batch_size, shuffle=True)

x, y = next(iter(train_loader))

# print(x.shape.y.shape,x.min(),y.min())
# print(x.shape.x.min())
plot_image(x, y, 'image sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(hw2+b2)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x


net = Net()
# lr 学习率 mom 动量
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28 * 28)

        out = net(x)

        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)

        loss = F.mse_loss(out, y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)

    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()

    total_correct += correct
total_sum = len(test_loader.dataset)

acc = total_correct / total_sum

print('test acc:', acc)

x, y = next(iter(test_loader))

out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
