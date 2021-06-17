import time

import torch
from torch import nn
import torchvision
from torchvision import transforms

class BasicBlock(nn.Module):

    def __init__(self, conv_module, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            conv_module(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_module(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv_module(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, conv_module, lin_module, block_module, num_blocks, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.layer1 = nn.Sequential(
            conv_module(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.layer2 = self._make_layer(conv_module, block_module, in_channels=64, out_channels=64, num_blocks=num_blocks[0], first_conv_stride=1)
        self.layer3 = self._make_layer(conv_module, block_module, in_channels=64, out_channels=128, num_blocks=num_blocks[1], first_conv_stride=2)
        self.layer4 = self._make_layer(conv_module, block_module, in_channels=128, out_channels=256, num_blocks=num_blocks[2], first_conv_stride=2)
        self.layer5 = self._make_layer(conv_module, block_module, in_channels=256, out_channels=512, num_blocks=num_blocks[3], first_conv_stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = lin_module(512, num_classes)

    # def _make_layer(self, block, out_channels, num_blocks, stride):
    #
    #     # we have num_block blocks per layer, the first block
    #     # could be 1 or 2, other blocks would always be 1
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels
    #
    #     return nn.Sequential(*layers)

    def _make_layer(self, conv_module, block_module, in_channels, out_channels, first_conv_stride, num_blocks):
        strides = [first_conv_stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block_module(conv_module, in_channels, out_channels, stride=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def ResNet18(conv_module, lin_module):
    return ResNet(conv_module, lin_module, BasicBlock, [2, 2, 2, 2])

def kaiming_initialize(net):
  for m in net.modules():
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      nn.init.kaiming_normal_(m.weight)

def get_cifar100_dataloader(train=True, mean=0, std=1, batch_size=128, num_workers=4, shuffle=True):
    transform_set = None
    if train:
        transform_set = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_set = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    data = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform_set)
    loader = torch.utils.data.DataLoader(data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return loader



# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
def n_right(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.sum(correct[:k].flatten()).float() #correct[:k].view(-1).float().sum(0)
        res.append(correct_k.item())
    return res


def train_epoch(net, optimizer, criterion, device, loader):
    net.train()

    epoch_loss = 0
    epoch_acc = 0
    epoch_acc_5 = 0

    for batch_index, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        y_hat = net(X)

        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        # _, preds = outputs.max(1)
        # correct += preds.eq(labels).sum()

        # if epoch <= 1: #args.warm
        #     warmup_scheduler.step()

        batch_loss = loss.item() * X.size(0)
        epoch_loss += batch_loss
        batch_accs = n_right(y_hat, y, topk=(1, 5))
        epoch_acc += batch_accs[0]
        epoch_acc_5 += batch_accs[1]

    epoch_loss /= len(loader.dataset)
    epoch_acc /= len(loader.dataset)
    epoch_acc_5 /= len(loader.dataset)
    return (epoch_loss, epoch_acc, epoch_acc_5)

@torch.no_grad()
def test_epoch(net, optimizer, criterion, device, loader):
    net.eval()

    epoch_loss = 0
    epoch_acc = 0
    epoch_acc_5 = 0

    for batch_index, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        y_hat = net(X)

        loss = criterion(y_hat, y)

        # _, preds = outputs.max(1)
        # correct += preds.eq(labels).sum()

        batch_loss = loss.item() * X.size(0)
        epoch_loss += batch_loss
        batch_accs = n_right(y_hat, y, topk=(1, 5))
        epoch_acc += batch_accs[0]
        epoch_acc_5 += batch_accs[1]

    epoch_loss /= len(loader.dataset)
    epoch_acc /= len(loader.dataset)
    epoch_acc_5 /= len(loader.dataset)
    return (epoch_loss, epoch_acc, epoch_acc_5)


def train_model(model, optimizer, train_scheduler, criterion, device, train_loader, test_loader,
                    n_epochs, print_every_n_epochs=1):
    # set up optimization metrics
    train_losses = []
    train_accs = []
    train_accs_5 = []
    test_losses = []
    test_accs = []
    test_accs_5 = []

    for epoch in range(n_epochs):
        e_start_t = time.time()

        if epoch > 1 and train_scheduler is not None:
            train_scheduler.step()

        # trainin'
        tr_loss, tr_acc, tr_acc_5 = train_epoch(model, optimizer, criterion, device, train_loader)
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        train_accs_5.append(tr_acc_5)

        # testin'
        te_loss, te_acc, te_acc_5 = test_epoch(model, optimizer, criterion, device, test_loader)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        test_accs_5.append(te_acc_5)

        if epoch % print_every_n_epochs == (print_every_n_epochs - 1):
            print(f'Epoch: {epoch}\t'
                  f'Epoch time: {time.time() - e_start_t:.2f} --- '
                  f'Train loss: {tr_loss:.4f}\t'
                  f'Test loss: {te_loss:.4f}\t'
                  f'Train accuracy: {100 * tr_acc:.2f}\t'
                  f'(top 5): {100 * tr_acc_5:.2f}\t'
                  f'Test accuracy: {100 * te_acc:.2f}\t'
                  f'(top 5): {100 * te_acc_5:.2f}')

    return (train_accs, train_accs_5, train_losses, test_accs, test_accs_5, test_losses)