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
import os


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
n_epochs = 10
learning_rate = 0.01
image_name = "real_images/easy_one.jpg"

net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
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
    net.train()
    for epoch in range(1, n_epochs+1):
        for data in train_loader:
            images, labels = data
            outputs = net(images) # This has an extra dimension, why? ([32, 2])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Training in %d" %epoch)

    for epoch in range(1, n_epochs + 1):
        valid_loss = 0.0
        net.eval()
        for data in valid_loader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            valid_loss = loss.item() * len(data)

        print("Loss during validation in the iteration %d equals: %f" % (epoch, valid_loss))



    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 on dummy faces images: %d %%' % (
        100 * correct / total))

    import cv2

    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    clone = cv2.imread(image_name)
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    winW = winH = 36
    faces = pyramid_sliding_window_detection(net, np.array(norm_image, dtype='float32'), 1.2, 36, 36, 5)

    total = 0
    for face in faces:
        face_array = np.array(face, dtype=int) # This is to convert the data to int, as it comes as float
        total += 1
        cv2.rectangle(clone, (face_array[0], face_array[1]), (face_array[2], face_array[3]), (255, 0, 0), 2)
    cv2.imshow("Tada", clone)
    path = '~/Documents/INSA/machine learning/git_Alan/ml_insa_lyon/result'
    cv2.imwrite(os.path.join(path, 'super_detected_best_ever_recognition.jpg'), clone)
    cv2.waitKey(0)
    print(total)

