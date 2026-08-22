# import all libraries in the code below
import flwr as fl
import sys

sys.path.append('..')
from metrics.computation import RAMU
from utils import get_parameters, set_parameters, train, test, predict, predict_gen
import pickle
import os
import copy
from CL.default import Memory
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import torch
import numpy as np


class FlowerClient_Root(fl.client.NumPyClient):
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
		return [val.cpu().numpy() for _, val in self.net.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.net.fc_module.train()
		return [val.cpu().numpy() for _, val in self.net.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.net.conv_module.train()
		params_dict = zip(self.net.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		self.net.fc_module.train()
		params_dict = zip(self.net.fc_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		ramu=RAMU()
		init_ram=ramu.compute("TRAINING")

		self.set_parameters(parameters)
		if config['server_round'] == 1:
			peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		
		if config['server_round'] > 1:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.net.fc_module.load_state_dict(state_dict, strict=True)
			peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		state_dict = self.net.fc_module.state_dict()
		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)  # OFFICEDB_MODIFICATIONS.md item 33
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
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
		loss, avg_pearson, avg_rmse, y_true, y_pred = test(self.net, self.testloader, self.y_labels, self.DEVICE)
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		# OFFICEDB_MODIFICATIONS.md item 33: raw predictions for CCC/CwM.
		np.savez(f'{self.path}/clientwise/preds{int(self.cid)}_round{config["server_round"]}.npz',
				 y_true=y_true, y_pred=y_pred, y_labels=np.array(self.y_labels))
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClientCL_Root(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params, n_tasks=2, active_idx_per_task=None, cumulative_idx_per_task=None, round_offset=0):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params, fedroot=True)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
		# See client/default.py:FlowerClientCL for n_tasks/rounds_per_task
		# (OFFICEDB_MODIFICATIONS.md).
		self.n_tasks = n_tasks
		self.rounds_per_task = self.nrounds // n_tasks
		# FCL Axis A masking -- see client/default.py:FlowerClientCL.
		self.active_idx_per_task = active_idx_per_task
		self.cumulative_idx_per_task = cumulative_idx_per_task
		# Chained-job resume (OFFICEDB_MODIFICATIONS.md item 25): 0 reproduces original behavior exactly.
		self.round_offset = round_offset

	def _task_idx(self, server_round):
		effective_round = server_round + self.round_offset
		return min((effective_round - 1) // self.rounds_per_task, self.n_tasks - 1)

	def _is_boundary(self, server_round, task_idx):
		effective_round = server_round + self.round_offset
		return (effective_round % self.rounds_per_task == 0) and (task_idx < self.n_tasks - 1)

	def get_parameters_all(self, config):
		print(f"[Client {self.cid}] get_parameters")
		# self.net.train()
		self.strat.model.train()
		return [val.cpu().numpy() for _, val in self.strat.model.state_dict().items()]
	

	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.conv_module.train()
		# self.net.conv_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters FC")
		self.strat.model.fc_module.train()
		# self.net.fc_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		print(f"[Client {self.cid}] set_parameters")
		self.strat.model.conv_module.train()
		# self.net.conv_module.train()
		params_dict = zip(self.strat.model.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		print(f"[Client {self.cid}] set_parameters FC")
		# self.net.fc_module.train()
		self.strat.model.fc_module.train()
		params_dict = zip(self.strat.model.fc_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		init_ram=RAMU().compute("TRAINING")

		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.strat.model.fc_module.load_state_dict(state_dict, strict=True)

			# set_parameters(self.net.fc_module, state_dict)
		except:
			print('')
		# self.strat.load_model(self.get_parameters_all({}))
		
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
		except:
			task_count = 0
		try:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				reg_term = pickle.load(f)
		except:
			reg_term = {}

		server_round = config["server_round"]
		task_idx = self._task_idx(server_round)
		train_loader = self.trainloaders[task_idx][int(self.cid)]
		if self.active_idx_per_task is not None:
			self.strat.active_idx = self.active_idx_per_task[task_idx]

		if self._is_boundary(server_round, task_idx):
			task_count += 1
			peak_ram = self.strat.learn_batch(task_count, reg_term, train_loader, learn_ewc=True)
			reg_term = self.strat.regularization_terms
		else:
			peak_ram = self.strat.learn_batch(task_count, reg_term, train_loader, learn_ewc=False)
			reg_term = self.strat.regularization_terms

		with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(reg_term, f)
		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		# state_dict = get_parameters(self.net.fc_module)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		state_dict = self.get_parameters_fc(config)

		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		return self.get_parameters(config={}), len(train_loader), {}

	def evaluate(self, parameters, config):
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		task_idx = self._task_idx(config["server_round"])
		eval_loader = self.valloader[task_idx]
		if self.cumulative_idx_per_task is not None:
			active_idx = self.cumulative_idx_per_task[task_idx]
			y_labels = [self.y_labels[i] for i in active_idx]
		else:
			active_idx, y_labels = None, self.y_labels
		loss, avg_pearson, avg_rmse, _, _ = test(self.strat.model, eval_loader, y_labels, self.DEVICE, active_idx=active_idx)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(eval_loader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClient_NR_Root(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params, n_tasks=2, active_idx_per_task=None, cumulative_idx_per_task=None, round_offset=0):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params, fedroot=True)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
		# See client/default.py:FlowerClientCL for n_tasks/rounds_per_task;
		# see client/default.py:FlowerClient_NR for the NR memory caveat
		# (OFFICEDB_MODIFICATIONS.md).
		self.n_tasks = n_tasks
		self.rounds_per_task = self.nrounds // n_tasks
		# FCL Axis A masking -- see client/default.py:FlowerClientCL.
		self.active_idx_per_task = active_idx_per_task
		self.cumulative_idx_per_task = cumulative_idx_per_task
		# Chained-job resume (OFFICEDB_MODIFICATIONS.md item 25): 0 reproduces original behavior exactly.
		self.round_offset = round_offset

	def _task_idx(self, server_round):
		effective_round = server_round + self.round_offset
		return min((effective_round - 1) // self.rounds_per_task, self.n_tasks - 1)

	def _is_boundary(self, server_round, task_idx):
		effective_round = server_round + self.round_offset
		return (effective_round % self.rounds_per_task == 0) and (task_idx < self.n_tasks - 1)

	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.conv_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.fc_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.strat.model.conv_module.train()
		params_dict = zip(self.strat.model.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		self.strat.model.fc_module.train()
		params_dict = zip(self.strat.model.fc_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		init_ram=RAMU().compute("TRAINING")

		# set_parameters(self.net, parameters)
		print(f"[Client {self.cid}] fit, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		# self.strat.model = copy.deepcopy(self.net)
		# print(config.keys())
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
		except:
			task_count = 0

		server_round = config["server_round"]
		effective_round = server_round + self.round_offset  # OFFICEDB_MODIFICATIONS.md item 25
		task_idx = self._task_idx(server_round)
		train_loader = self.trainloaders[task_idx][int(self.cid)]
		is_boundary = self._is_boundary(server_round, task_idx)
		if self.active_idx_per_task is not None:
			self.strat.active_idx = self.active_idx_per_task[task_idx]

		if task_idx == 0 and not is_boundary:
			peak_ram = self.strat.learn_batch(task_count, {}, train_loader, False)
		elif is_boundary:
			task_count += 1
			peak_ram = self.strat.learn_batch(task_count, {}, train_loader, True)

			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(self.strat.task_memory[task_count].storage, f)
		else:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				stor = pickle.load(f)
			memory = {task_count: Memory()}
			memory[task_count].update(stor)
			peak_ram = self.strat.learn_batch(task_count, memory, train_loader, False)
			if effective_round == int(self.nrounds):
				if os.path.exists(f'{self.path}/reg{int(self.cid)}.pkl'):
					os.remove(f'{self.path}/reg{int(self.cid)}.pkl')

		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		# state_dict = get_parameters(self.net.fc_module)
		state_dict = self.get_parameters_fc(config)

		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		return self.get_parameters(config={}), len(train_loader), {}

	def evaluate(self, parameters, config):
		# check if {self.path}/{self.num_clients} exists, if not create it
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		task_idx = self._task_idx(config["server_round"])
		eval_loader = self.valloader[task_idx]
		if self.cumulative_idx_per_task is not None:
			active_idx = self.cumulative_idx_per_task[task_idx]
			y_labels = [self.y_labels[i] for i in active_idx]
		else:
			active_idx, y_labels = None, self.y_labels
		loss, avg_pearson, avg_rmse, _, _ = test(self.strat.model, eval_loader, y_labels, self.DEVICE, active_idx=active_idx)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(eval_loader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClient_LGR(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params, gr, n_tasks=2, active_idx_per_task=None, cumulative_idx_per_task=None, round_offset=0):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params, gr, path, cid)
		self.gr=gr
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
		# See client/default.py:FlowerClientCL for n_tasks/rounds_per_task
		# (OFFICEDB_MODIFICATIONS.md). Unlike the EWC-style clients, LGR
		# increments task_count AFTER calling learn_batch at a boundary (not
		# before) -- LatentGenerativeReplay.learn_batch branches on
		# self.task_count==0 to decide "first task" vs. "replay", and the
		# boundary round is still that completing task's last round.
		self.n_tasks = n_tasks
		self.rounds_per_task = self.nrounds // n_tasks
		# FCL Axis A masking -- see client/default.py:FlowerClientCL.
		self.active_idx_per_task = active_idx_per_task
		self.cumulative_idx_per_task = cumulative_idx_per_task
		# Chained-job resume (OFFICEDB_MODIFICATIONS.md item 25): 0 reproduces original behavior exactly.
		self.round_offset = round_offset

	def _task_idx(self, server_round):
		effective_round = server_round + self.round_offset
		return min((effective_round - 1) // self.rounds_per_task, self.n_tasks - 1)

	def _is_boundary(self, server_round, task_idx):
		effective_round = server_round + self.round_offset
		return (effective_round % self.rounds_per_task == 0) and (task_idx < self.n_tasks - 1)

	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.conv_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.fc_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.strat.model.conv_module.train()
		params_dict = zip(self.strat.model.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, state_dict) -> None:
		self.strat.model.fc_module.train()
		# params_dict = zip(self.net.fc_module.state_dict().keys(), parameters)
		# state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		init_ram=RAMU().compute("TRAINING")

		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.set_parameters_fc(state_dict)

			# set_parameters(self.net.fc_module, state_dict)
		except:
			print('')
		# self.strat.model = copy.deepcopy(self.net)
		
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
			print("----------task counted---------")
		except:
			task_count = 0
		try:
			with open(f'{self.path}/gen{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				reg_term = pickle.load(f)
			self.gr.load_state_dict(state_dict, strict=True)
		except:
			reg_term=self.gr.state_dict()

		server_round = config["server_round"]
		task_idx = self._task_idx(server_round)
		train_loader = self.trainloaders[task_idx][int(self.cid)]
		is_boundary = self._is_boundary(server_round, task_idx)
		if self.active_idx_per_task is not None:
			self.strat.active_idx = self.active_idx_per_task[task_idx]

		if is_boundary:
			# Boundary round is still the completing task's last round of
			# training -- task_count increments AFTER, not before (see
			# __init__ docstring above).
			peak_ram=self.strat.learn_batch(task_count, reg_term, train_loader, learn_gen=True)
			reg_term = self.strat.get_generator_weights()
			task_count += 1
		else:
			peak_ram=self.strat.learn_batch(task_count, reg_term, train_loader, learn_gen=False)
			reg_term = self.strat.get_generator_weights()
		with open(f'{self.path}/gen{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(reg_term, f)
		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		# state_dict = get_parameters(self.net.fc_module)
		state_dict = self.get_parameters_fc(config)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		return self.get_parameters(config={}), len(train_loader), {}

	def evaluate(self, parameters, config):
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		task_idx = self._task_idx(config["server_round"])
		eval_loader = self.valloader[task_idx]
		if self.cumulative_idx_per_task is not None:
			active_idx = self.cumulative_idx_per_task[task_idx]
			y_labels = [self.y_labels[i] for i in active_idx]
		else:
			active_idx, y_labels = None, self.y_labels
		loss, avg_pearson, avg_rmse, _, _ = test(self.strat.model, eval_loader, y_labels, self.DEVICE, active_idx=active_idx)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(eval_loader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}

