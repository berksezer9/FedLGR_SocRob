# import all libraries in the code below
# import OrderedDict, List, np, torch
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
# import ordereddict
from collections import OrderedDict
import numpy as np
import flwr as fl
import pickle
import sys
import os

sys.path.append('..')
from metrics.computation import RAMU
from utils import get_parameters, set_parameters_bn, train, test, predict, predict_gen


class FlowerClient_BN(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, testloader, epochs, y_labels, num_clients, DEVICE, path):
		self.cid = cid
		self.net = net
		self.trainloader = trainloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.path = path
		self.num_clients = num_clients
		self.DEVICE = DEVICE
	
	def get_parameters_bn(self, config):
		print(f"[Client {self.cid}] get_parameters_bn for local client.")
		# self.net.train()
		return [val.cpu().numpy() for name, val in self.net.state_dict().items() if "bn" in name]
	
	def get_parameters(self, config) -> List[np.ndarray]:
		print(f"[Client {self.cid}] get_parameters for client.")
		# Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
		self.net.train()
		return [val.cpu().numpy() for name, val in self.net.state_dict().items() if "bn" not in name]
	
	def set_parameters_bn(self, parameters: List[np.ndarray]) -> None:
		# self.net.train()
		# Set net parameters from a list of NumPy ndarrays
		keys = [k for k in self.net.state_dict().keys() if "bn" in k]
		params_dict = zip(keys, parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.load_state_dict(state_dict, strict=False)
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.net.train()
		# Set net parameters from a list of NumPy ndarrays
		keys = [k for k in self.net.state_dict().keys() if "bn" not in k]
		params_dict = zip(keys, parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.load_state_dict(state_dict, strict=False)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		init_ram=RAMU().compute('Train')
		self.set_parameters(parameters)
		# try and load bn parameters if they exist, use if statement so that it crashes if they exist but are not the right shape
		# if exist then load, else train
		if config['server_round'] > 1:
			with open(f'{self.path}/mod_bn{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.set_parameters_bn(state_dict)
		peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		
		# get and save bn parameters
		state_dict = self.get_parameters_bn(config)
		with open(f'{self.path}/mod_bn{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)  # OFFICEDB_MODIFICATIONS.md item 33
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		return self.get_parameters(config=config), len(self.trainloader), {}

	def evaluate(self, parameters, config):
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)
		self.set_parameters(parameters)
		if config['server_round'] > 1:
			with open(f'{self.path}/mod_bn{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.set_parameters_bn(state_dict)
		loss, avg_pearson, avg_rmse, avg_ccc, y_true, y_pred = test(self.net, self.testloader, self.y_labels, self.DEVICE)

		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse},{avg_ccc}\n')
		# OFFICEDB_MODIFICATIONS.md item 33: raw predictions for CwM (needs raw
		# per-annotator ratings not available here); item 34: avg_ccc logged above.
		# item 35: sample_id lets y_true/y_pred be joined back to any raw
		# annotation data post-hoc without retraining/re-inferring.
		np.savez(f'{self.path}/clientwise/preds{int(self.cid)}_round{config["server_round"]}.npz',
				 y_true=y_true, y_pred=y_pred, y_labels=np.array(self.y_labels),
				 sample_id=np.array(getattr(self.testloader.dataset, 'sample_ids', [])))
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse, "avg_ccc": avg_ccc}


class FlowerClient_BN_Root(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, testloader, epochs, y_labels, num_clients, DEVICE, path):
		self.cid = cid
		self.net = net
		self.trainloader = trainloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.path = path
		self.num_clients = num_clients
		self.DEVICE = DEVICE
	
	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.net.conv_module.train()
		return [val.cpu().numpy() for name, val in self.net.conv_module.state_dict().items() if "bn" not in name]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.net.fc_module.train()
		return [val.cpu().numpy() for name, val in self.net.fc_module.state_dict().items() if "bn" not in name]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.net.conv_module.train()
		# Set net parameters from a list of NumPy ndarrays
		keys = [k for k in self.net.conv_module.state_dict().keys() if "bn" not in k]
		params_dict = zip(keys, parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.conv_module.load_state_dict(state_dict, strict=False)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		self.net.fc_module.train()
		# Set net parameters from a list of NumPy ndarrays
		keys = [k for k in self.net.fc_module.state_dict().keys() if "bn" not in k]
		params_dict = zip(keys, parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.fc_module.load_state_dict(state_dict, strict=False)
	
	def fit(self, parameters, config):
		init_ram=RAMU().compute('Train')
		print(f"[Client {self.cid}] fit, config: {config}")
		self.set_parameters(parameters)
		if config['server_round'] == 1:
			peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		
		if config['server_round'] > 1:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.net.fc_module.load_state_dict(state_dict, strict=True)
			# self.set_parameters_fc(state_dict)
			peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		state_dict = self.net.fc_module.state_dict()
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)  # OFFICEDB_MODIFICATIONS.md item 33
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		return self.get_parameters(config={}), len(self.trainloader), {}

	def evaluate(self, parameters, config):
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)
		print(f"[Client {self.cid}] evaluate, config: {config}")
		self.set_parameters(parameters)

		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.net.fc_module.load_state_dict(state_dict, strict=True)
		except:
			print('')
		loss, avg_pearson, avg_rmse, avg_ccc, y_true, y_pred = test(self.net, self.testloader, self.y_labels, self.DEVICE)
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse},{avg_ccc}\n')
		# OFFICEDB_MODIFICATIONS.md item 33: raw predictions for CwM (needs raw
		# per-annotator ratings not available here); item 34: avg_ccc logged above.
		# item 35: sample_id lets y_true/y_pred be joined back to any raw
		# annotation data post-hoc without retraining/re-inferring.
		np.savez(f'{self.path}/clientwise/preds{int(self.cid)}_round{config["server_round"]}.npz',
				 y_true=y_true, y_pred=y_pred, y_labels=np.array(self.y_labels),
				 sample_id=np.array(getattr(self.testloader.dataset, 'sample_ids', [])))
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse, "avg_ccc": avg_ccc}
