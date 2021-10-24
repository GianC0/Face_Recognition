#!/bin/env python3


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from net import Net


class Trainer:

    def __init__(self,train_dir,batch_size,valid_size=0,n_epochs=1,learning_rate=0.01,device=torch.device('cpu')):

        self.device=device
        self.learning_rate=learning_rate
        self.n_epochs=n_epochs
        self.valid_size=valid_size
        self.batch_size=batch_size
        self.train_dir = train_dir

    def get_trained_network(self):
        transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0),std=(1))])

        train_data = torchvision.datasets.ImageFolder(self.train_dir, transform=transform)

        net = Net().to(self.device)
        optimizer = optim.SGD(net.parameters(), self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        num_train = len(train_data)
        indices_train = list(range(num_train))
        np.random.shuffle(indices_train)
        split_tv = int(np.floor(self.valid_size * num_train))
        train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

        # useful if there's cuda self.device
        kwargs={'num_workers': 1, 'pin_memory': True} if self.device.type == 'cuda' else{}

        train_sampler = SubsetRandomSampler(train_new_idx)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler, **kwargs)

        # IN CASE OF MULTIPLE MODELS COMPARISON
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=valid_sampler, **kwargs)

        # classes = ('noface','face')

        for epoch in range(1, self.n_epochs+1):
            epoch_loss = 0  # useful for calculating avg loss at each epoch
            c = 0  # counter for all the values in the train_loader. Useful for avg loss

            for images,labels in train_loader:

                # loading data into GPU
                images=images.to(self.device)
                labels=labels.to(self.device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                c+=1  # incrementing the counter

            print('epoch:', epoch,'\ttotal loss:', epoch_loss/c*100)  # printing statistics  EPOCH-AVG_LOSS
        return net



