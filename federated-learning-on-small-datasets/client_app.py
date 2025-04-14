
import os
import torch
import random
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord
from test_app.task1 import Net, get_weights, load_data, set_weights, test, train, fix_random
import tomli
from torchvision import transforms



with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)

# Extract parameters
parameters = config.get("tool", {}).get("CustomConfig", {})

# Access values
the_device = parameters.get("device")
BASE_SEED= parameters.get("seed")


os.environ["CUDA_VISIBLE_DEVICES"] = f"{the_device}"

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader,  local_epochs):    
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") 
        self.net.to(self.device)

       

    def fit(self, parameters, config):
        """Train a model using as a starting point the parameters sent by the ServerApp.
        Then, communicate the weights of the locally-updated model back to the
        ServerApp.
        """
        BASE_SEED =2

        current_round = config.get("current_round")

        fix_random(BASE_SEED + current_round)
        # Apply parameters to local model
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
        )

        return (
            get_weights(self.net),  # Return parameters of the locally-updated model
            len(
                self.trainloader.dataset
            ),  # Training examples used (needed sometimes for aggregation)
            {
                "train_loss": train_loss,
            },  # Communicate metrics
        )



def client_fn(context: Context):
    """A function that returns a Client."""

    # Instantiate the model
    net = Net()
    # Read node config and fetch data for the ClientApp that is being constructed
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader = load_data(partition_id, num_partitions)

    # Read the run config (defined in the `pyproject.toml`)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)