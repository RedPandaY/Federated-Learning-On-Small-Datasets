import random
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import torch
import json
import wandb
from datetime import datetime
from flwr.common import FitIns, Context
from test_app.task1 import fix_random
import tomli
from torchvision import transforms
from typing import List, Tuple
import numpy as np

from functools import partial, reduce
from typing import Any, Callable, Union


from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from logging import WARNING
from typing import Callable, Optional, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

# Load pyproject.toml
with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)
# Extract parameters
parameters = config.get("tool", {}).get("CustomConfig", {})

BASE_SEED=parameters.get("seed")

"""def aggregate_simple(results: list[tuple[NDArrays, int]]) -> NDArrays:
        
        
        num_clients = len(results)  # Total number of clients

        # Create a list of only the weights, ignoring the number of examples
        weights = [weights for weights, _ in results]

        # Compute the simple (unweighted) average of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_clients  # Divide by number of clients
            for layer_updates in zip(*weights)
        ]
        return weights_prime"""

class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}
       # self.inplace= False

        # Log those same metrics to W&B
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-Dirichlit0.05", name=f"FedAvg_lr=0,1,SGD-{name}")

    
        

    """ def aggregate_simple_inplace(
    results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        
        num_clients = len(results)
        scaling_factor = 1.0 / num_clients

        # Use the first client's parameters as the accumulator.
        # We assume that "results[0][0]" is a list of NumPy arrays.
        aggregated = results[0][0]
        # Scale the first client's parameters in place.
        for i in range(len(aggregated)):
            aggregated[i] *= scaling_factor

        # For each remaining client, add their parameters (scaled equally) in place.
        for client_params, _ in results[1:]:
            for i in range(len(aggregated)):
                aggregated[i] += client_params[i] * scaling_factor

        return aggregated"""

    """def aggregate_fitu(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
     ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate_simple(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated  """
        

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
       # ndarrays = parameters_to_ndarrays(parameters_aggregated)
        # Instantiate model
       # model = Net()
        # Apply paramters to model
        #set_weights(model, ndarrays)
        # Save global model in the standard PyTorch 
        #if server_round >=9999:
        # torch.save(model.state_dict(), f"global_model_round_lr=01{server_round}")

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
        with open("results_lr=0.1.json", "w") as json_file:
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


class FedAvgWithDC(FedAvg):
    """
    Federated Daisy-Chaining (FedDC) Strategy:
      - Daisy-chaining period (d): redistributes client models in a chain.
      - Aggregation period (b): performs standard FedAvg aggregation.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        daisy_period : int
            Daisy-chaining period d.
        agg_period : int
            Aggregation period b.
        kwargs : dict
            Other parameters to forward to FedAvg's constructor (e.g., fraction_fit).
        """
        super().__init__(**kwargs)
        #self.inplace= False
        # Log to Weights & Biases
        run_name = datetime.now().strftime("FedAvgWithDC(init_lr=0.1)SGD_%Y%m%d_%H%M%S")
        wandb.init(project="flower-Dirichlit0.05", name=run_name)

        # Store client parameters for daisy-chaining.
        # This dictionary will eventually hold entries for all 250 clients.
        self.last_round_parameters = {}
        # For saving metrics.
        self.results_to_save = {}
        # Global model parameters.
        self.global_params = None
        # Will store all client IDs once we see the full client list.
        self.all_client_ids = None

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Assigns the correct parameters to each client depending on the round type:
          - Aggregation round: broadcast the new global model.
          - Daisy-chaining round: exchange client-specific parameters.
          - Other rounds: instructions maintain a sequential order.
        """
        fix_random(BASE_SEED+server_round)
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # On the first call, store all client IDs.
        if self.all_client_ids is None:
            all_clients = client_manager.all() 
            self.all_client_ids = list(all_clients.keys())
        
        # Sample clients for this round.
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # --- Aggregation Round ---
        # We use an aggregation round when server_round % b == 0 
        # because the previous aggregation happened in round (b - 1) and updated the global model.
        if ((server_round % b) == (0)) or (server_round == 1):
            print("Aggregation round: sending new global params to clients")
            instructions = super().configure_fit(server_round, parameters, client_manager)
            return instructions
        
        # --- Daisy-Chaining Round ---
        if (server_round % d) == (d - 1):
            print("Daisy-chaining round: shuffling clients")
            # Create a shuffled list of indices for model assignment.
            client_indices = list(range(len(clients)))
            random.shuffle(client_indices)

            daisy_instructions = []
            for original_idx, assigned_idx in zip(range(len(clients)), client_indices):
                sender_cid = clients[assigned_idx].cid  # Sender (randomly chosen from sample)
                receiver_proxy = clients[original_idx]  # Receiver
                # Use stored parameters if available, otherwise fall back to the current global parameters.
                prev_params = self.last_round_parameters.get(sender_cid)
                fit_ins = FitIns(prev_params, config)
                daisy_instructions.append((receiver_proxy, fit_ins))
            return daisy_instructions
        else:
            # Or we could just increase number local epochs until the next b or d round!
            print("Training instructions without shuffling")
            client_indices = list(range(len(clients)))
            
            instructions = []
            for original_idx, assigned_idx in zip(range(len(clients)), client_indices):
                sender_cid = clients[assigned_idx].cid  
                receiver_proxy = clients[original_idx] 
                prev_params = self.last_round_parameters.get(sender_cid)
                fit_ins = FitIns(prev_params, config)
                instructions.append((receiver_proxy, fit_ins))
            return instructions    
            

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate results based on round type:
          - Aggregation round: perform FedAvg aggregation and update stored parameters for all clients.
          - Non-aggregation (daisy-chaining) round: update stored parameters only for the sampled clients.
        """
        # --- Aggregation Round ---
        if (server_round % b) == (b - 1):
            print("Aggregation round: updating global parameters")
            aggregated_params, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
            self.global_params = aggregated_params

            if aggregated_params is not None:
              print("# Update last_round_parameters for all clients.")
              for cid in self.all_client_ids:
                self.last_round_parameters[cid] = aggregated_params

            return aggregated_params, metrics_aggregated
        
        # --- Non-Aggregation Rounds ---
        # if paramters are not found add them otherwise
        # update only the parameters of the sampled clients,
        # while preserving parameters for unsampled clients.
        print("Non-aggregation round: returning latest global params")
        for client_proxy, fit_res in results:
           self.last_round_parameters[client_proxy.cid] = fit_res.parameters
        
        return self.global_params, {}

    def aggregate_fitu(self, server_round, results, failures):
        
            # --- Aggregation Round ---
            if (server_round % b) == (b - 1):
                print("Aggregation round: updating global parameters")
                """Aggregate fit results using weighted average."""
                if not results:
                    return None, {}
                # Do not aggregate if there are failures and failures are not accepted
                if not self.accept_failures and failures:
                    return None, {}

                if self.inplace:
                    # Does in-place weighted average of results
                    aggregated_ndarrays = aggregate_inplace(results)
                else:
                    # Convert results
                    weights_results = [
                        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                        for _, fit_res in results
                    ]
                    aggregated_ndarrays = aggregate_simple(weights_results)

                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

                # Aggregate custom metrics if aggregation fn was provided
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")







                self.global_params = parameters_aggregated

                if parameters_aggregated is not None:
                    print("# Update last_round_parameters for all clients.")
                    for cid in self.all_client_ids:
                        self.last_round_parameters[cid] = parameters_aggregated

                return parameters_aggregated, metrics_aggregated
        
            # --- Non-Aggregation Rounds ---
            # if paramters are not found add them otherwise
            # update only the parameters of the sampled clients,
            # while preserving parameters for unsampled clients.
            print("Non-aggregation round: returning latest global params")
            for client_proxy, fit_res in results:
                self.last_round_parameters[client_proxy.cid] = fit_res.parameters
            
            return self.global_params, {}    

    

    def evaluate(self, server_round, parameters):
        """
        Evaluate the global model and log metrics to JSON and Weights & Biases.
        """
        loss, metrics = super().evaluate(server_round, parameters)
        self.results_to_save[server_round] = {"loss": loss, **metrics}
        with open("results_feddc(init_b10).json", "w") as outfile:
            json.dump(self.results_to_save, outfile, indent=4)
        wandb.log({"round_eval": server_round, "loss": loss, **metrics}, step=server_round)
        return loss, metrics

class BaselineDC(FedAvg):
    """
    Federated Daisy-Chaining (FedDC) Strategy:
      - Daisy-chaining period (d): redistributes client models in a chain.
      - Aggregation period (b): performs standard FedAvg aggregation.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        daisy_period : int
            Daisy-chaining period d.
        agg_period : int
            Aggregation period b.
        kwargs : dict
            Other parameters to forward to FedAvg's constructor (e.g., fraction_fit).
        """
        super().__init__(**kwargs)

        # Log to Weights & Biases
        run_name = datetime.now().strftime("BaselineDC_%Y%m%d_%H%M%S")
        wandb.init(project="flower-Dirichlit0.05", name=run_name)

        # Store client parameters for daisy-chaining.
        # This dictionary will eventually hold entries for all 250 clients.
        self.last_round_parameters = {}
        # For saving metrics.
        self.results_to_save = {}
        # Global model parameters.
        self.global_params = None
        # Will store all client IDs once we see the full client list.
        self.all_client_ids = None
        

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Assigns the correct parameters to each client depending on the round type:
          - Aggregation round: broadcast the new global model.
          - Daisy-chaining round: exchange client-specific parameters.
          - Other rounds: (if needed) fallback to default behavior.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # On the first call, store all client IDs.
        if self.all_client_ids is None:
            all_clients = client_manager.all() 
            self.all_client_ids = list(all_clients.keys())

      
        # Sample clients for this round.
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
      
        if (server_round == 1):
           instructions = super().configure_fit(server_round, parameters, client_manager)
           return instructions
       
        # --- Daisy-Chaining Round ---
        if (server_round % d) == (d - 1):
            print("Daisy-chaining round: shuffling clients")
            # Create a shuffled list of indices for model assignment.
            client_indices = list(range(len(clients)))
            random.shuffle(client_indices)

            daisy_instructions = []
            for original_idx, assigned_idx in zip(range(len(clients)), client_indices):
                sender_cid = clients[assigned_idx].cid  # Sender (randomly chosen from sample)
                receiver_proxy = clients[original_idx]  # Receiver
                # Use stored parameters if available, otherwise fall back to the current global parameters.
                prev_params = self.last_round_parameters.get(sender_cid)
                fit_ins = FitIns(prev_params, config)
                daisy_instructions.append((receiver_proxy, fit_ins))
            return daisy_instructions


    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate results based on round type:
          - Aggregation round: perform FedAvg aggregation and update stored parameters for all clients.
          - Non-aggregation (daisy-chaining) round: update stored parameters only for the sampled clients.
        """
        for client_proxy, fit_res in results:
            self.last_round_parameters[client_proxy.cid] = fit_res.parameters
        # --- Aggregation Round ---
        if (server_round % b) == (b - 1) or (server_round == 1):
            print("Aggregation round: updating global parameters")   
            aggregated_params, metrics_aggregated = super().aggregate_fit(server_round, results, failures)     

            self.global_params = aggregated_params
            return self.global_params, {}

        print("Non-aggregation round: returning latest global params")
        # Return the global parameters (which were set during the last aggregation round).
        return self.global_params, {}

    def evaluate(self, server_round, parameters):
        """
        Evaluate the global model and log metrics to JSON and Weights & Biases.
        """
        loss, metrics = super().evaluate(server_round, parameters)
        self.results_to_save[server_round] = {"loss": loss, **metrics}
        with open("results_feddc(init_b10).json", "w") as outfile:
            json.dump(self.results_to_save, outfile, indent=4)
        wandb.log({"round_eval": server_round, "loss": loss, **metrics}, step=server_round)
        return loss, metrics
