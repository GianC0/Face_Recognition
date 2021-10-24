import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import trainer
import time
from config import config
from pyramid import pyramid_sliding_window_detection
import os
import cv2
from datetime import datetime
import numpy as np






# Measuring time
start = time.time()

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







transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0), std=(1))])

device = torch.device('cpu')


kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}
test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

trainer=trainer.Trainer(train_dir,batch_size,valid_size,n_epochs,learning_rate,device)
net=trainer.get_trained_network()


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # loading data
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 on dummy faces  images: %d %%' % (100 * correct / total))
print("\n \t\t\t\t DURATION: ", time.time()-start)







'''
# Saving output in folders
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

        print("-> Beginning pyramid algorithm")
        faces = pyramid_sliding_window_detection(net, np.array(norm_image, dtype='float32'), 1.2, 36, 36, 5,device)

        print("-> Counting faces", end = " ...")
        total = 0
        for face in faces:
            face_array = np.array(face, dtype=int)  # This is to convert the data to int, as it comes as float
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

'''