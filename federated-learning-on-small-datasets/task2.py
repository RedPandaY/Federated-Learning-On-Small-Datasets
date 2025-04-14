from flamby.datasets.fed_ixi import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    SEEDS,
    Baseline,
    BaselineLoss,
    FedIXITiny,
    Optimizer,
    metric,
)
from flamby.utils import evaluate_model_on_tests

# test_app/task.py
from collections import OrderedDict
from flwr_datasets import FederatedDataset
#from test_app.task import get_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from flwr_datasets.partitioner import DirichletPartitioner,SizePartitioner, IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Subset
import random
from datasets import Dataset
from torch.utils.data import Subset




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

def load_data(partition_id: int, num_partitions=33):
    if 0 <= partition_id < 249:
        dataset = FedIXITiny(
            center=0,
            train=True,
            pooled=False,
        )
        subdatasets = partition_dataset(dataset, [1] * 249)
        
        generator = torch.Generator()
        generator.manual_seed(0)
        
        partition = subdatasets[partition_id]
        trainloader = DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        return trainloader

    if partition_id >= 249 and partition_id < 394:
        dataset = FedIXITiny(
            center=1,
            train=True,
            pooled=False,
        )
        subdatasets = partition_dataset(dataset, [1] * 145)
        generator = torch.Generator()
        generator.manual_seed(0)
        
        # Adjust index for center 1
        partition = subdatasets[partition_id - 249]
        trainloader = DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        return trainloader

    if partition_id >= 394 and partition_id < 453:
        dataset = FedIXITiny(
            center=2,
            train=True,
            pooled=False,
        )
        subdatasets = partition_dataset(dataset, [1] * 59)
        generator = torch.Generator()
        generator.manual_seed(0)
        
        # Adjust index for center 2
        partition = subdatasets[partition_id - 394]
        trainloader = DataLoader(partition, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        return trainloader




  
def train(model, trainloader, epochs, lr, device):
        """
        Train the model using Flamby's default loss and optimizer.
        
        Parameters:
          model: The neural network model (e.g., the Baseline model).
          trainloader: DataLoader for training data.
          epochs: Number of training epochs.
          lr: Learning rate.
        
        Returns:
          float: The average training loss.
        """
        model.to(device)
        model.train()
        criterion = BaselineLoss()
        optimizer = Optimizer(model.parameters(), lr=lr)
        running_loss = 0.0
        counter = 0
        for _ in range(epochs):
            for batch in trainloader:
                # FedIXITiny returns (img, label)
                images = batch["image"] if isinstance(batch, dict) else batch[0]
                labels = batch["label"] if isinstance(batch, dict) else batch[1]
                #images, labels = batch
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if counter == 100: 
                  break
        avg_loss = running_loss / len(trainloader)
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



