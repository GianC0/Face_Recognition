#!/bin/env python3
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from pyramid_default import pyramid_sliding_window_detection
from net import Net


train_dir = './train_images'
test_dir = './test_images'
real_dir = './real_images'

transform = transforms.Compose(
    [transforms.Grayscale(), 
     transforms.ToTensor(), 
     transforms.Normalize(mean=(0,),std=(1,))])

train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

valid_size = 0.05
batch_size = 32
n_epochs = 20

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_train = len(train_data)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:],indices_train[:split_tv]

train_sampler = SubsetRandomSampler(train_new_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=1)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)
classes = ('noface','face')

if __name__ == "__main__":
    min_valid_loss = 0.5
    for epoch in range(1, n_epochs+1):
        net.train()
        for data in train_loader:
            optimizer.zero_grad()
            images, labels = data
            outputs = net(images) # This has an extra dimension, why? ([32, 2])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        valid_loss = 0.0
        net.eval()
        for data in valid_loader:
            optimizer.zero_grad()
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            valid_loss = loss.item() * len(data)

        print("Loss during validation in the iteration %d equals: %f" % (epoch, valid_loss))
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), 'saved_model.pth')





    correct = 0
    total = 0

    with torch.no_grad():
        net_saved = Net()
        net_saved.load_state_dict(torch.load('saved_model.pth'))
        net_saved.eval()
        for data in test_loader:
            images, labels = data
            outputs = net_saved(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 on dummy faces images: %d %%' % (
        100 * correct / total))

    import cv2

    image = cv2.imread("real_images/slowdive.jpg", cv2.IMREAD_GRAYSCALE)
    winW = winH = 36
    scales = pyramid_sliding_window_detection(net_saved, np.array(image, dtype='float32'), 1.2, 36, 36, 5)

    clone = image.copy()
    # The shape of the output of pyramid_sliding_... is [scale] where scale is [scale, [face]] and face is
    # [startx, starty, endx, endy]
    total = 0
    for scale in scales:
        for face in scale[1]:
            face = np.array(face, dtype=int)
            total += 1
            cv2.rectangle(clone, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 2)
            # break
        # break
    print(total)
    cv2.imshow("Tada", clone)
    cv2.waitKey(0)

