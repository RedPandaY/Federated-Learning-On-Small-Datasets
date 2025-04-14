import random
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import torch
import json
import wandb
from datetime import datetime
from flwr.common import FitIns
from test_app.task1 import Net, set_weights,get_weights
import tomli



class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        # Log those same metrics to W&B
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-tutorial", name=f"FedAvg_b=10-{name}")

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and metrics, ave global model checkpoint."""

        # Call the default aggregate_fit method from FedAvg
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        ## Save new Global Model as a PyTorch checkpoint
        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        # Instantiate model
        model = Net()
        # Apply paramters to model
        set_weights(model, ndarrays)
        # Save global model in the standard PyTorch 
        if server_round >=999:
         torch.save(model.state_dict(), f"global_model_round_10{server_round}")

        # Return the expected outputs for `aggregate_fit`
        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model, then save metrics to local JSON and to W&B."""
        # Call the default behaviour from FedAvg
        loss, metrics = super().evaluate(server_round, parameters)

        # Store metrics as dictionary
        my_results = {"loss": loss, **metrics}
        # Insert into local dictionary
        self.results_to_save[server_round] = my_results

        # Save metrics as json
        with open("results_b=10.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss, metrics
    
with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)

# Extract parameters
parameters = config.get("tool", {}).get("CustomConfig", {})

# Access values
b = parameters.get("b")
d = parameters.get("d")
import numpy as np


class FedAvgWithDC(FedAvg):
    """
    Federated Daisy-Chaining (FedDC) Strategy:
      - Daisy-chaining period (d): redistributes client models in a chain.
      - Aggregation period (b): performs standard FedAvg aggregation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Log to Weights & Biases
        run_name = datetime.now().strftime("FedAvgWithDC(1)_%Y%m%d_%H%M%S")
        wandb.init(project="flower-simulation", name=run_name)
        # Store client parameters for (if needed) daisy-chaining.
        self.last_round_parameters = {}
        self.results_to_save = {}
        self.global_params = None
        self.all_client_ids = None

def configure_fit(self, server_round, parameters, client_manager):
    config = {}
    if self.on_fit_config_fn is not None:
        config = self.on_fit_config_fn(server_round)
        
    # On the first call, store all client IDs.
    if self.all_client_ids is None:
        all_clients = client_manager.all() 
        self.all_client_ids = list(all_clients.keys())
        
    sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
    clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
    
    # Initialize global parameters on round 1.
    if server_round == 1:
        self.global_params = parameters

    # --- Aggregation Round ---
    # For example, if b=10, we perform aggregation when server_round % b == 9.
    if (server_round % b) == (0):
        print("Aggregation round: sending new global params to clients")
        instructions = super().configure_fit(server_round, parameters, client_manager)
        return instructions

    # --- Daisy-Chaining Round (Dataset Shuffling) ---
    elif (server_round % d) == (d - 1):
        print("Daisy-chaining round: instructing clients to reshuffle dataset")
        # Instead of exchanging model parameters, send the current global params with a flag.
        fit_ins = FitIns(self.global_params, {**config, "shuffle_data": True})
        instructions = [(client, fit_ins) for client in clients]
        return instructions

    # --- Fallback: use current global parameters.
    else:
        print("Fallback round: using global parameters")
        fit_ins = FitIns(self.global_params, config)
        instructions = [(client, fit_ins) for client in clients]
        return instructions

def aggregate_fit(self, server_round, results, failures):
    # --- Aggregation Round ---
    if (server_round % b) == (b - 1):
        print("Aggregation round: updating global parameters")
        aggregated_params, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is not None:
            if self.all_client_ids is not None:
                for cid in self.all_client_ids:
                    self.last_round_parameters[cid] = aggregated_params
            else:
                for client_proxy, _ in results:
                    self.last_round_parameters[client_proxy.cid] = aggregated_params
        self.global_params = aggregated_params
        ndarrays = parameters_to_ndarrays(aggregated_params)
        model = Net()
        set_weights(model, ndarrays)
        if server_round >= 9999:
            torch.save(model.state_dict(), f"global_model_round_DC(4){server_round}")
        return aggregated_params, metrics_aggregated
    else:
        # In non-aggregation rounds, no parameter exchange is performed.
        print("Non-aggregation round: returning global parameters")
        return self.global_params, {}

    def evaluate(self, server_round, parameters):
        """
        Evaluate the global model and log metrics to JSON and Weights & Biases.
        """
        loss, metrics = super().evaluate(server_round, parameters)
        self.results_to_save[server_round] = {"loss": loss, **metrics}
        with open("results_feddc(4).json", "w") as outfile:
            json.dump(self.results_to_save, outfile, indent=4)
        wandb.log({"round_eval": server_round, "loss": loss, **metrics}, step=server_round)
        return loss, metrics
