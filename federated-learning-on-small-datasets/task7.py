import albumentations
from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
)
from flamby.utils import check_dataset_from_config, evaluate_model_on_tests

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

def load_data(partition_id: int, num_partitions: int):
    """
    Loads a DataLoader corresponding to a client partition for the ISIC 2019 dataset.
    
    Each center (0 to 5) is partitioned into 10 clients, each with 8 samples.
    Global partition IDs 0 to 59 are mapped such that:
      - center = partition_id // 10
      - local_client_id = partition_id % 10

    Parameters:
      partition_id (int): Global client ID (should be in the range 0 to 59).
      num_partitions (int): This parameter is kept for consistency but is not used.
    
    Returns:
      DataLoader: A DataLoader for the corresponding client's data.
    """
    # Define constants: 10 clients per center, each with 8 samples.
    clients_per_center = 10
    samples_per_client = 8
    num_centers = 6  # ISIC 2019 has 6 centers
    total_clients = num_centers * clients_per_center
    
    # Determine which center and which client in that center
    center = partition_id // clients_per_center
    local_client_id = partition_id % clients_per_center

    # Load the dataset corresponding to the center
    dataset = FedIsic2019(center=center, train=True)
    
    # Partition the dataset into 10 subsets, each with 8 samples
    partition_sizes = [samples_per_client] * clients_per_center
    subdatasets = partition_dataset(dataset, partition_sizes)

    # Create a DataLoader for the selected partition
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(subdatasets[local_client_id], batch_size=8, shuffle=True, generator=generator)




  
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
        # not like the original
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        running_loss = 0.0
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



