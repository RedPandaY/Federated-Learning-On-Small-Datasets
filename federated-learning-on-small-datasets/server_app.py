"""test-app: A Flower / PyTorch app."""

from typing import List
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from ray import _config
from torchvision import transforms
from test_app.task1 import Net, get_transform, get_weights,set_weights, test, fix_random 
#from test_app.task7 import Net, get_weights,set_weights, test, fix_random 
from flwr_datasets.partitioner import DirichletPartitioner,SizePartitioner, IidPartitioner
from test_app.strategies import CustomFedAvg, FedAvgWithDC, BaselineDC
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from flwr.server.strategy import FedAvg
import random
import tomli
from torchvision import transforms
from torchvision.transforms import Compose, Resize, RandomApply, RandomHorizontalFlip, RandomAffine, ToTensor, Normalize
import numpy as np


"""from flamby.datasets.fed_ixi import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    SEEDS,
    Baseline,
    BaselineLoss,
    FedIXITiny,
    Optimizer,
    metric,
 )"""
"""from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)"""
"""from flamby.datasets.fed_isic2019 import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedIsic2019,
    metric,
)"""

strategy_map = {
    "CustomFedAvg": CustomFedAvg,
    "FedAvgWithDC": FedAvgWithDC,
    "FedAvg": FedAvg,
    "BaselineDC": BaselineDC,
}

# Load pyproject.toml
with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)

# Extract parameters
parameters = config.get("tool", {}).get("CustomConfig", {})

# Access values
#b = parameters.get("b")

BASE_SEED=parameters.get("seed")
if True:
 print("SEED WORKING")  
 fix_random(BASE_SEED)
def get_evaluate_fn_dice(test_dataloaders, metric, device):
    """
    Return a callback that evaluates the global model using the Dice coefficient.
    
    Parameters:
        test_dataloaders (List[torch.utils.data.DataLoader]):
            List of test dataloaders (one per client or test partition).
        dice_metric (callable):
            Function with signature (y_true: np.ndarray, y_pred: np.ndarray) -> scalar
            that computes the Dice score.
        device (torch.device):
            Device to run evaluation on (e.g. torch.device("cuda") or torch.device("cpu")).
    
    Returns:
        A function that takes (server_round, parameters_ndarrays, config) and returns:
            (loss, {"cen_dice": average_dice_score})
        where loss is defined as 1 - average_dice_score.
    """
    def evaluate(server_round, parameters_ndarrays, config):
        # Instantiate the model and set its weights
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        
        # Use the provided test function that works with a list of dataloaders and a metric
        dice_scores = test(net, test_dataloaders, metric)
        
        # Compute the average dice score over all test dataloaders
        if dice_scores:
            avg_dice = sum(dice_scores.values()) / len(dice_scores)
        else:
            avg_dice = 0.0
        
        # Define loss as 1 - dice so that minimizing loss corresponds to a higher Dice score
        loss = 1.0 - avg_dice
        
        return loss, {"cen_dice": avg_dice}
    
    return evaluate

# server-side evaluation (Global evaluation)
def get_evaluate_fn(testloader, device):
   """ return a callback that evaluates the global model"""
    
   def evaluate(server_round, parameters_ndarrays, config):# how is server_round getting passed? !!
      """Evaluate global model using provided centralised testset."""
      net = Net()
      set_weights(net, parameters_ndarrays)
      net.to(device)
      loss, accuracy = test(net, testloader, device)

      return loss,{"cen_accuracy": accuracy}
   
   return evaluate

# server-side evaluation (Global evaluation)
def get_evaluateH_fn(testloader, metric, device):
   """ return a callback that evaluates the global model"""
    
   def evaluate(server_round, parameters_ndarrays, config):
      """Evaluate global model using provided centralised testset."""
      net = Net()
      set_weights(net, parameters_ndarrays)
      net.to(device)
      accuracy = test(net, testloader, metric)

      return 0,{"cen_accuracy": accuracy}
   
   return evaluate   
def get_evaluate_fn_DC(testloader, device):
    """Return a callback that evaluates the global model every b rounds."""
    
    last_loss, last_accuracy = 0.0, 0.0  # Store last valid results

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate the global model every b rounds."""
        nonlocal last_loss, last_accuracy  # Keep track of the last evaluation

        if server_round % b == b - 1:  
            net = Net()
            set_weights(net, parameters_ndarrays)
            net.to(device)
            loss, accuracy = test(net, testloader, device)

            # Update last known values
            last_loss, last_accuracy = loss, accuracy

            return loss, {"cen_accuracy": accuracy}

        # Return last known values to keep W&B plots smooth
        return last_loss, {"cen_accuracy": last_accuracy}

    return evaluate


# aggregate evaluation metrics from different clients
def weighted_average(metrics): 
    """ A function, that apply weighted AVG aggregation """

    accuracies = [num_examples* m["accuracy"] for num_examples, m in metrics]
    total_examples= sum(num_examples for num_examples, _ in metrics)
    
    return {"accuracy": sum(accuracies)/ total_examples}
def simple_average(metrics): 
    """A function to calculate the unweighted average (simple mean) of accuracies."""
    
    accuracies = [m["accuracy"] for _, m in metrics]  # Extract accuracies from the metrics
    return {"accuracy": sum(accuracies) / len(accuracies) if accuracies else 0}

# determine how to aggregate training metrics reported by clients during the fit (training) phase
def handle_fit_matrics(metrics): 
   """ retrun the training loss of each client """
   for _,m in metrics:
      print(m) 
   return {}

# provide configuration parameters to clients before the training phase begins   
def on_fit_config(server_round: int):
    """ Adjust learning rate based on server round """
    
    # Initial learning rate
    #lr = LR
    lr = 0.1
    # Define schedule: Reduce LR by half every 2500 rounds
    sched_rounds = 2500
    lr *= 0.5 ** (server_round // sched_rounds)

    return {"lr": lr, "current_round": server_round}

"""def get_test_transform():
    pytorch_transforms = Compose([
         transforms.Lambda(lambda img: img.convert("RGB")),
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch
    
    return apply_transforms"""

def server_fn(context: Context):
    vald_map = {
     "get_evaluate_fn_DC": get_evaluate_fn_DC,
     "get_evaluate_fn": get_evaluate_fn,
     "get_evaluate_fn_dice": get_evaluate_fn_dice,
     "get_evaluateH_fn":get_evaluateH_fn
    }

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction_evaluate"]
    min_available_clients = context.run_config["min_available_clients"]
    strategy_class = strategy_map.get(context.run_config["strategy"])
    global_vald_fn = vald_map.get(context.run_config["global_vald_fn"])

    # Initialize model parameters
    #ndarrays = get_weights(Net())
    #parameters = ndarrays_to_parameters(ndarrays)
    model = Net()
    checkpoint_path = "global_model_round_9453493"  # Update with latest saved model

    # Load the checkpoint if it exists
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print("No saved model found. Starting training from scratch.")

    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)
    
    #Loading global test set
    testset= load_dataset("uoft-cs/cifar10")["test"]

    testloader= DataLoader(testset.with_transform(get_transform()), batch_size=64)

    #dataset_dict = load_dataset("imagefolder", data_dir= "mri")
    #dataset = dataset_dict["test"]
    #partitioner = SizePartitioner([len(dataset)])
        
    #partitioner.dataset = dataset    
    #partition = partitioner.load_partition(0)
    #partition = partition.with_transform(get_transform())
    #testloader = DataLoader(partition, batch_size=8)

    """ testloader = DataLoader(
        FedIsic2019(train=False, pooled=True),
        batch_size=8,
        shuffle=False,
      )"""
   
    """testloader = DataLoader(
        FedIXITiny(train=False, pooled=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
     )"""
    #Define strategy
    strategy = strategy_class(
        
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=parameters,
        #evaluate_metrics_aggregation_fn=simple_average,
        #fit_metrics_aggregation_fn=handle_fit_matrics,
        on_fit_config_fn=on_fit_config,
        evaluate_fn= global_vald_fn(testloader,device=("cuda")),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
