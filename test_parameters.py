#!/bin/env python3
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from net import Net, NetExtraLinear, NetExtraConv
import time

if __name__ == '__main__':
    train_dir = './data/train_images'
    test_dir = './data/test_images'
    real_dir = './data/real_images'

    transform = transforms.Compose(
        [transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0,), std=(1,))])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    valid_size = 0.3
    batch_size = 32
    net_set = ["Net", "NetExtraLinear"]

    criterion = nn.CrossEntropyLoss()

    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
    classes = ('noface', 'face')

    n_epochs_set = [25, 20]
    results = []
    n = [1, 2]
    for number in n:

        for n_epochs in n_epochs_set:
            for net_string in net_set:
                if net_string == "Net":
                    net = Net()
                elif net_string == "NetExtraLinear":
                    net = NetExtraLinear()
                start = time.time()
                optimizer = optim.SGD(net.parameters(), lr=0.01)
                print("nmbr epoch: ")
                for epoch in range(1, n_epochs + 1):
                    print(f" {epoch}", end="")
                    for data in train_loader:
                        optimizer.zero_grad()
                        images, labels = data
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                print(" fin")
                end_train = time.time()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                end_test = time.time()

                print(f"{n_epochs} + {net_string}:")
                print('Accuracy of the network on the 10000 on dummy faces  images: %d %%' % (
                        100 * correct / total))
                print(f"TimeTrain: {end_train - start} TimeOverall: {end_test - start}")
                results.append(
                    [n_epochs, net_string, 100 * correct / total, end_train - start, end_test - start])

    with open('results/results.txt', 'w') as testfile:
        for row in results:
            testfile.write(' '.join([str(a) for a in row]) + '\n')
