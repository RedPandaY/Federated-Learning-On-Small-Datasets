from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)
from flamby.utils import evaluate_model_on_tests

# test_app/task.py
from collections import OrderedDict
from flwr_datasets import FederatedDataset
from test_app.task import get_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner,SizePartitioner, IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import random




def fix_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
fix_random(0)
Net = Baseline
def partition_dataset(dataset, partition_sizes):
    """
    Partition a PyTorch dataset into multiple subdatasets based on the provided partition sizes.

    Parameters:
      dataset (Dataset): A torch.utils.data.Dataset instance.
      partition_sizes (List[int]): A list of integers representing the number of samples for each partition.
      
    Returns:
      List[Subset]: A list of torch.utils.data.Subset instances, each corresponding to one partition.
    """
    total = len(dataset)
    if sum(partition_sizes) > total:
        raise ValueError("Sum of partition sizes is greater than dataset size.")
    
    indices = list(range(total))
    partitions = []
    start = 0
    for size in partition_sizes:
        end = start + size
        partitions.append(Subset(dataset, indices[start:end]))
        start = end
    return partitions
"""def load_data(partition_id: int, num_partitions: int):   
      dataset = FedHeartDisease(center=partition_id,train=True)
      generator = torch.Generator()
      generator.manual_seed(0)  
      return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,generator=generator)"""
def load_data(partition_id: int, num_partitions: int):
    # Define the partition sizes for each center
    cleveland_parts = 199
    hungary_parts   = 172
    switzerland_parts = 30
    long_beach_parts  = 85

    generator = torch.Generator()
    generator.manual_seed(0)

    # First group: Cleveland
    if 0 <= partition_id < cleveland_parts:
        dataset = FedHeartDisease(center=0, train=True)
        subdatasets = partition_dataset(dataset, [1] * cleveland_parts)
        partition = subdatasets[partition_id]
        return DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    # Second group: Hungary
    elif partition_id < cleveland_parts + hungary_parts:
        dataset = FedHeartDisease(center=1, train=True)
        subdatasets = partition_dataset(dataset, [1] * hungary_parts)
        # Adjust partition_id for the offset of Cleveland partitions
        partition = subdatasets[partition_id - cleveland_parts]
        return DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    # Third group: Switzerland
    elif partition_id < cleveland_parts + hungary_parts + switzerland_parts:
        dataset = FedHeartDisease(center=2, train=True)
        subdatasets = partition_dataset(dataset, [1] * switzerland_parts)
        # Adjust for the previous two groups
        partition = subdatasets[partition_id - (cleveland_parts + hungary_parts)]
        return DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    # Fourth group: Long Beach V
    elif partition_id < cleveland_parts + hungary_parts + switzerland_parts + long_beach_parts:
        dataset = FedHeartDisease(center=3, train=True)
        subdatasets = partition_dataset(dataset, [1] * long_beach_parts)
        partition = subdatasets[partition_id - (cleveland_parts + hungary_parts + switzerland_parts)]
        return DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    
    else:
        raise ValueError("Invalid partition_id")


  
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Assume BaselineLoss and evaluate_model_on_tests are imported from your library.
# from flamby.datasets.fed_heart_disease import BaselineLoss
# from flamby.utils import evaluate_model_on_tests

def train(model: torch.nn.Module, trainloader, epochs: int, lr: float, device: torch.device):
    """
    Train the model using Flamby's default loss and optimizer.
    
    Parameters:
      model: The neural network model.
      trainloader: DataLoader for training data.
      epochs: Number of training epochs.
      lr: Learning rate.
      device: Device to train on (CPU or GPU).
    
    Returns:
      float: The average training loss.
    """
    model.to(device)
    model.train()
    criterion = BaselineLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  
    running_loss = 0.0
    
    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                        
    avg_loss = running_loss / (len(trainloader) * epochs)
  
    return avg_loss

def test(model, testloader, metric):
   return evaluate_model_on_tests(model,[testloader],metric)

    
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



