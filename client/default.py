# import all libraries in the code below
import flwr as fl
import sys

sys.path.append('..')
from metrics.computation import RAMU
from CL.default import Memory
import numpy as np
import pickle
import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
# import ordereddict
from collections import OrderedDict
import flwr as fl
import sys
from utils import get_parameters, set_parameters, train, train_scaffold, test, predict, predict_gen


class FlowerClient(fl.client.NumPyClient):
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
	
	def get_parameters(self, config) -> List[np.ndarray]:
		# print(f"[Client {self.cid}] get_parameters")
		self.net.train()
		return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.net.train()
		params_dict = zip(self.net.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		ramu = RAMU()
		print(f"[Client {self.cid}] fit, config: {config}")
		self.set_parameters(parameters)
		init_ram = ramu.compute("TRAINING")
		peak_ram = train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		# write it in a file
		# exist_ok=True (OFFICEDB_MODIFICATIONS.md item 33): 3 Ray actors racing
		# to create the same directory intermittently hit FileExistsError under
		# the plain exists-then-makedirs check (observed job 34058073, SCAFFOLD
		# k5_test4/seed1 -- lost 1 client's round-1 fit to this).
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)
		with open(f'{self.path}/clientwise/ramu{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram - init_ram}\n')
		# local_steps: total local SGD steps this round (epochs * batches) --
		# FedNovaStrategy.aggregate_fit (server/strategies.py) needs this to
		# normalize each client's update by how much local computation it
		# actually did (OFFICEDB_MODIFICATIONS.md item 29). Additive-only: every
		# other strategy either ignores unknown fit-metrics keys entirely or
		# only aggregates them through fit_metrics_aggregation_fn, which stays
		# unset (None) everywhere except FedNovaStrategy, so this is a no-op
		# for FedAvg/FedProx/FedOptAdam/FedBN/FedDistill/FedRoot.
		local_steps = self.epochs * len(self.trainloader)
		return self.get_parameters(config={}), len(self.trainloader), {"local_steps": local_steps}
	
	def evaluate(self, parameters, config):
		os.makedirs(f'{self.path}/clientwise', exist_ok=True)
		self.set_parameters(parameters)
		loss, avg_pearson, avg_rmse, avg_ccc, y_true, y_pred = test(self.net, self.testloader, self.y_labels, DEVICE=self.DEVICE)
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse},{avg_ccc}\n')
		# preds{cid}_round{r}.npz (OFFICEDB_MODIFICATIONS.md item 33): raw
		# per-scene (y_true, y_pred) pairs for this client/round, so CwM
		# (eval/metrics.py, needs raw per-annotator ratings not available
		# here) can be computed later without retraining. avg_ccc (item 34)
		# is already logged above -- CCC only needs (y_true, y_pred).
		# sample_id (item 35): the per-scene Stamp for each row, in the same
		# order -- lets any post-hoc metric (e.g. CwM, or a per-action
		# breakdown) join back to raw annotation data without needing to
		# reproduce the dataloader's row order, which load_datasets()'s
		# unseeded data_images_test.sample(frac=1) reshuffle otherwise makes
		# unsafe to assume across a fresh process.
		np.savez(f'{self.path}/clientwise/preds{int(self.cid)}_round{config["server_round"]}.npz',
				 y_true=y_true, y_pred=y_pred, y_labels=np.array(self.y_labels),
				 sample_id=np.array(getattr(self.testloader.dataset, 'sample_ids', [])))
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse, "avg_ccc": avg_ccc}


class FlowerClientScaffold(FlowerClient):
	"""SCAFFOLD (Karimireddy et al., ICML 2020) client: identical to
	FlowerClient except for fit() -- the `parameters` this receives from
	SCAFFOLDStrategy.configure_fit (server/strategies.py) are [model_weights,
	correction] concatenated (correction = global_c - client_c for this
	client, computed and owned server-side -- see that class's docstring for
	why control-variate state isn't persisted client-side the way the CL
	clients' reg{cid}.pkl/task{cid}.txt is). get_parameters/set_parameters
	are inherited unchanged since they only ever see plain model weights (the
	`weights` half, already split out below, before set_parameters is
	called). See OFFICEDB_MODIFICATIONS.md item 31.
	"""

	def fit(self, parameters, config):
		ramu = RAMU()
		print(f"[Client {self.cid}] fit, config: {config}")
		n_params = len(parameters) // 2
		weights, correction = parameters[:n_params], parameters[n_params:]
		self.set_parameters(weights)
		init_ram = ramu.compute("TRAINING")
		peak_ram = train_scaffold(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE,
		                           correction=correction)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ramu{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram - init_ram}\n')
		local_steps = self.epochs * len(self.trainloader)
		return self.get_parameters(config={}), len(self.trainloader), {"local_steps": local_steps}


class FlowerClientCL(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params, n_tasks=2, active_idx_per_task=None, cumulative_idx_per_task=None, round_offset=0):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
		# n_tasks=2 reproduces the original hardcoded circle/arrow 2-task split
		# (rounds_per_task == nrounds/2); OfficeDB's Axis A/B pass n_tasks=3/6
		# (see OFFICEDB_MODIFICATIONS.md).
		self.n_tasks = n_tasks
		self.rounds_per_task = self.nrounds // n_tasks
		# FCL Axis A (action-subset) masking, both optional/None by default
		# (reproduces unmasked behaviour exactly): active_idx_per_task[t] is
		# this task's OWN exclusive active columns (used for training, set on
		# self.strat before learn_batch); cumulative_idx_per_task[t] is the
		# union of columns revealed by tasks 0..t (used for evaluation, per
		# the design doc's "train exclusive, evaluate cumulative" protocol).
		# See OFFICEDB_MODIFICATIONS.md.
		self.active_idx_per_task = active_idx_per_task
		self.cumulative_idx_per_task = cumulative_idx_per_task
		# Chained-job resume (OFFICEDB_MODIFICATIONS.md item 25): 0 reproduces original behavior exactly.
		self.round_offset = round_offset

	def _task_idx(self, server_round):
		effective_round = server_round + self.round_offset
		return min((effective_round - 1) // self.rounds_per_task, self.n_tasks - 1)

	def _is_boundary(self, server_round, task_idx):
		# Last round of a task (except the last task, which has no further
		# task to prepare importance/replay for).
		effective_round = server_round + self.round_offset
		return (effective_round % self.rounds_per_task == 0) and (task_idx < self.n_tasks - 1)

	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		# self.net.train()
		self.strat.model.train()
		return [val.cpu().numpy() for _, val in self.strat.model.state_dict().items()]

	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		# self.net.train()
		print(f"[Client {self.cid}] set_parameters")
		self.strat.model.train()
		params_dict = zip(self.strat.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.load_state_dict(state_dict, strict=True)

	def fit(self, parameters, config):
		ramu = RAMU()
		init_ram = ramu.compute("Train")
		print(f"[Client {self.cid}] fit, config: {config}")
		self.set_parameters(parameters)
		# self.strat.load_model(parameters)
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
			peak_ramu = self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term, train_loader=train_loader,
			                                   learn_ewc=True)

			reg_term = self.strat.regularization_terms
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(reg_term, f)
			with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
				f.write(f'{task_count}')
		else:
			peak_ramu = self.strat.learn_batch(task_count=task_count, regularization_terms=reg_term, train_loader=train_loader,
			                                   learn_ewc=False)

		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ramu{int(self.cid)}.csv', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{init_ram},{peak_ramu},{peak_ramu - init_ram}\n')
		return self.get_parameters(config={}), len(train_loader), {}

	def evaluate(self, parameters, config):
		# check if {self.path}/{self.num_clients} exists, if not create it
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')

		self.set_parameters(parameters)
		# self.strat.load_model(parameters)
		# self.valloader is the cumulative per-task-boundary test loader list
		# (tasks 0..task_idx combined) -- see task_splitter in dataloader/utils.py.
		task_idx = self._task_idx(config["server_round"])
		eval_loader = self.valloader[task_idx]
		if self.cumulative_idx_per_task is not None:
			active_idx = self.cumulative_idx_per_task[task_idx]
			y_labels = [self.y_labels[i] for i in active_idx]
		else:
			active_idx, y_labels = None, self.y_labels
		loss, avg_pearson, avg_rmse, _, _, _ = test(self.strat.model, eval_loader, y_labels, self.DEVICE, active_idx=active_idx)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(eval_loader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClient_NR(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
	             strat_name, params, n_tasks=2, active_idx_per_task=None, cumulative_idx_per_task=None, round_offset=0):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
		# See FlowerClientCL for n_tasks/rounds_per_task (OFFICEDB_MODIFICATIONS.md).
		# NOTE: NR's reg{cid}.pkl only ever stores the immediately-preceding
		# task's memory (vendor's own original design, not something this
		# generalization changes) -- for n_tasks>2 this means rehearsal only
		# ever replays the most recent prior task, not the full task history.
		self.n_tasks = n_tasks
		self.rounds_per_task = self.nrounds // n_tasks
		# FCL Axis A masking -- see FlowerClientCL.
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
		# self.net.train()
		self.strat.model.train()
		return [val.cpu().numpy() for _, val in self.strat.model.state_dict().items()]

	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		print(f"[Client {self.cid}] set_parameters")
		# self.net.train()
		self.strat.model.train()
		params_dict = zip(self.strat.model.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.load_state_dict(state_dict, strict=True)

	def fit(self, parameters, config):
		ramu = RAMU()
		init_ram = ramu.compute("Train")
		print(f"[Client {self.cid}] fit, config: {config}")

		self.set_parameters(parameters)
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
			peak_ram = self.strat.learn_batch(task_count, {}, train_loader, learn_nr=False)

		elif is_boundary:
			task_count += 1
			peak_ram = self.strat.learn_batch(task_count, {}, train_loader, learn_nr=True)

			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(self.strat.task_memory[task_count].storage, f)
		else:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				storage = pickle.load(f)
			memory = {task_count: Memory()}
			memory[task_count].update(storage)
			peak_ram = self.strat.learn_batch(task_count, memory, train_loader, learn_nr=False)
			if effective_round == int(self.nrounds):
				if os.path.exists(f'{self.path}/reg{int(self.cid)}.pkl'):
					os.remove(f'{self.path}/reg{int(self.cid)}.pkl')
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ramu{int(self.cid)}.csv', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram - init_ram}\n')

		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		return self.get_parameters(config={}), len(train_loader), {}

	def evaluate(self, parameters, config):
		# check if {self.path}/{self.num_clients} exists, if not create it
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')

		# set_parameters(self.strat.model, parameters)
		self.set_parameters(parameters)
		task_idx = self._task_idx(config["server_round"])
		eval_loader = self.valloader[task_idx]
		if self.cumulative_idx_per_task is not None:
			active_idx = self.cumulative_idx_per_task[task_idx]
			y_labels = [self.y_labels[i] for i in active_idx]
		else:
			active_idx, y_labels = None, self.y_labels
		loss, avg_pearson, avg_rmse, _, _, _ = test(self.strat.model, eval_loader, y_labels, self.DEVICE, active_idx=active_idx)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(eval_loader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
