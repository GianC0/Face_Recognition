import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from pyramid import pyramid_sliding_window_detection
from net import Net
from config import config
from datetime import datetime
import os
import cv2


# Settings for the script
train_dir = config["dirs"]["train"]
test_dir = config["dirs"]["test"]
real_dir = config["dirs"]["real"]
valid_size = config["training"]["valid_size"]
batch_size = config["training"]["batch_size"]
n_epochs = config["training"]["n_epochs"]
learning_rate = config["training"]["learning_rate"]
input_path = config["usage"]["input"]
output_format = config["usage"]["output"]

# Transformations applied to the set
transform = transforms.Compose([
    transforms.Grayscale(), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0,),std=(1,))
    ])

# Steps needed to start the training
train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
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

print("Beginning training")
net.train() # Which is the function to stop the training?
for epoch in range(1, n_epochs+1):
    print(f"-> Epoch number {epoch}", end = " ...")
    for data in train_loader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(" DONE")

"""
# This is innecesary, as its not upgrading nothing (the gradient remains the same)
print("Beginning validation")
for epoch in range(1, n_epochs + 1):
    print(f"-> Epoch number {epoch}", end = " ...")
    valid_loss = 0.0
    net.eval()
    for data in valid_loader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        valid_loss = loss.item() * len(data)
    print(f"Loss: {valid_loss}")
"""
net.eval()
with torch.no_grad():
    print("Beginning testing")
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"-> Accuracy of the network on the 10000 dummy faces images: {100 * correct / total} %%")

if os.path.isdir(input_path):
    dt_string = datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss")
    output_path = "results/" + output_format + " " + dt_string
    print(f"Creating {output_path} folder")
    os.makedirs(output_path, exist_ok = True)

    for filename in os.listdir(input_path):
        file = os.path.join(input_path, filename)
        if os.path.isdir(file):
            continue

        print(f"Working with {file}")
        image_grayscale = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image_color = cv2.imread(file)
        norm_image = cv2.normalize(image_grayscale, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        winW = winH = 36

        print("-> Beginning pyramid trainning algorithm")
        faces = pyramid_sliding_window_detection(net, np.array(norm_image, dtype='float32'), 1.2, 36, 36, 5)

        print("-> Counting faces", end = " ...")
        total = 0
        for face in faces:
            face_array = np.array(face, dtype=int) # This is to convert the data to int, as it comes as float
            total += 1
            cv2.rectangle(image_color, (face_array[0], face_array[1]), (face_array[2], face_array[3]), (255, 0, 0), 2)
        print(f" Total faces recognized in image: {total}")

        filename, _ = os.path.splitext(os.path.basename(file))
        _, extension = os.path.splitext(file)
        output_file = f"{filename}.ml{extension}"
        print(f"-> {output_file} created")
        cv2.imwrite(os.path.join(output_path, output_file), image_color)

else:
    print("ERROR: problems with outputs or inputs.")
    print("Check config.py and validate the corresponding values")
    exit(1)