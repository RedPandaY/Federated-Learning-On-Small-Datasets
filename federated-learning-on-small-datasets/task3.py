from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import cv2
import torchvision
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner,SizePartitioner, IidPartitioner
from torchvision.transforms import Compose, Resize, RandomApply, RandomHorizontalFlip, RandomAffine, ToTensor, Normalize
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import random


def fix_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_random(0)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 3,padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.Flatten = nn.Flatten()
        # self.fc1 = nn.Linear(32*37*37,1024)
        # self.fc2 = nn.Linear(1024,2)
        self.fc = nn.Linear(64*37*37,2)

        #self.initialize_network()
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),2,stride = 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2,stride = 2)
        x = self.Flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc(x)
        
        return x

    def initialize_network(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
###################################################################################### 


def get_transform(size=(150,150)):
    def apply_transforms(batch):
        fix_random(0)
        transformed_images = []
        for img in batch["image"]:
            # Ensure the image is a numpy array (in case it’s not)
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            # Convert from BGR to RGB using cv2
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize the image to the desired size
            img = cv2.resize(img, size)
            # Normalize pixel values to [0,1]
            img = img.astype(np.float32) / 255.0
            # Change shape from (H, W, C) to (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            # Convert to torch tensor
            tensor = torch.tensor(img)
            transformed_images.append(tensor)
        batch["image"] = transformed_images
        return batch
    return apply_transforms

dataset_dict = None
def load_data(partition_id: int, num_partitions: int):
   # print("number of partitions: ")
   # print(num_partitions)
    global dataset_dict
    if dataset_dict is None:
     dataset_dict = load_dataset("imagefolder", data_dir= "mri")
    dataset = dataset_dict["train"]
    partitioner = IidPartitioner(num_partitions)
    partitioner.dataset = dataset    
    generator = torch.Generator()
    generator.manual_seed(0) 
    partition = partitioner.load_partition(partition_id)
    partition = partition.with_transform(get_transform())
    trainloader = DataLoader(partition, batch_size=8, shuffle=True, generator=generator)

    return trainloader   

###################################
# Training and evaluation functions
###################################
def train(model, trainloader, epochs, lr, device):
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay= 0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            labels = batch["label"] if isinstance(batch, dict) else batch[1]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_loss = running_loss / len(trainloader)
    return avg_loss

def test(model, testloader, device):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"] if isinstance(batch, dict) else batch[0]
            labels = batch["label"] if isinstance(batch, dict) else batch[1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    return avg_loss, accuracy

#############################
# Utility functions for Flower
#############################
def get_weights(model):
    # Return a list of NumPy arrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict}) 
    net.load_state_dict(state_dict, strict=True)



