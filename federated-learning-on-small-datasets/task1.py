# test_app/task.py
from collections import OrderedDict
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import random
from flwr_datasets.partitioner import DirichletPartitioner,SizePartitioner, IidPartitioner,ExponentialPartitioner,PathologicalPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Subset
import tomli


# Load pyproject.toml
with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)

# Extract parameters
parameters = config.get("tool", {}).get("CustomConfig", {})
BASE_SEED=parameters.get("seed")
def fix_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_random(BASE_SEED)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out
#############################
# ResNet for CIFAR-10 (ResNet18)
#############################
class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def __str__(self):
        return "ResNET"
# Cifar10ResNet18
class Net(ResNet):
    def __init__(self):
        ResNet.__init__(self, 18, 10)
    
################################################################
def get_transform():
     
    pytorch_transforms = Compose( 
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    
    return apply_transforms        

#############################
# Data transforms and loader
#############################
"""
fds = None
def load_data(partition_id: int, num_partitions: int):
   # print("number of partitions: ")
   # print(num_partitions)
    
    global fds
    if fds is None:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_classes_per_partition=2,
            class_assignment_mode="random",
            shuffle = True,
            seed = BASE_SEED,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    generator = torch.Generator()
    generator.manual_seed(BASE_SEED)      
    partition = fds.load_partition(partition_id)
    partition = partition.with_transform(get_transform())
    trainloader = DataLoader(partition, batch_size=64, shuffle=True,generator=generator)

    return trainloader"""

#############################
# Data transforms and loader
#############################

"""fds = None
def load_data(partition_id: int, num_partitions: int):
   # print("number of partitions: ")
   # print(num_partitions)
    
    global fds
    if fds is None:
        partitioner = ExponentialPartitioner(num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    generator = torch.Generator()
    generator.manual_seed(BASE_SEED)      
    partition = fds.load_partition(partition_id)
    partition = partition.with_transform(get_transform())
    trainloader = DataLoader(partition, batch_size=64, shuffle=True,generator=generator)

    return trainloader  """

#############################
# Data transforms and loader
#############################

"""fds = None
def load_data(partition_id: int, num_partitions: int):
   # print("number of partitions: ")
   # print(num_partitions)
    
    global fds
    if fds is None:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_classes_per_partition=2,
            class_assignment_mode="random",
            shuffle = True,
            seed = BASE_SEED
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
   # generator = torch.Generator()
    #generator.manual_seed(2)      
    partition = fds.load_partition(partition_id)
    partition = partition.with_transform(get_transform())
    trainloader = DataLoader(partition, batch_size=64, shuffle=True)

    return trainloader"""
#############################
# Data transforms and loader
############################# 
sized_fd= None   
def load_sized_partition():

  global sized_fd
  if sized_fd is None:
        sized_fd = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": SizePartitioner([9600])},
        )
        
  sized_partition = sized_fd.load_partition(0)
  return sized_partition
fds = None
def load_data(partition_id: int, num_partitions: int):
    sized_dataset =load_sized_partition()   
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                   alpha=0.5, min_partition_size=1,
                                   self_balancing=True)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
       
    fds._dataset = {"train": sized_dataset}
    fds._dataset_prepared = True    
    generator = torch.Generator()
    generator.manual_seed(BASE_SEED)      
    partition = fds.load_partition(partition_id)
    partition = partition.with_transform(get_transform())
    trainloader = DataLoader(partition, batch_size=64, shuffle=True,generator=generator)

    return trainloader  

#############################
# Training and evaluation functions
#############################
def train(model, trainloader, epochs, lr, device):
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"] if isinstance(batch, dict) else batch[0]
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
            images = batch["img"] if isinstance(batch, dict) else batch[0]
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

