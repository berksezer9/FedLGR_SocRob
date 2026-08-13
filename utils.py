from collections import OrderedDict
import os
import pickle
from typing import Dict, List, Optional, Tuple

import copy
import numpy
import torch
import torch.optim as optim
import torch.nn as nn
import math
import numpy as np
from scipy.stats import pearsonr
from dataloader.outloader import CustomOutputDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="darkgrid")

import flwr as fl
from flwr.common import NDArrays, Scalar
from metrics.computation import RAMU


class RMSELoss(torch.nn.Module):
	def __init__(self):
		super(RMSELoss, self).__init__()
	
	def forward(self, x, y):
		criterion = nn.MSELoss()
		loss = torch.sqrt(criterion(x, y))
		return loss


def truncate_float(float_number, decimal_places):
	# NaN/inf guard: a degenerate round (e.g. a per-task test split small
	# enough that a batch's Pearson correlation is undefined even after
	# test()'s own jitter-retry fallback -- more likely with OfficeDB's
	# smaller per-task-per-client splits than MANNERS-DB's original scale)
	# used to crash `int(nan * multiplier)` with ValueError here, killing an
	# entire multi-strategy/reg-coef sweep over one bad round. Passing
	# NaN/inf through unchanged (instead of truncating, which is meaningless
	# for them anyway) keeps the run alive and the value visibly NaN in the
	# results CSV, rather than silently swallowing it into a valid-looking
	# number or fatally crashing.
	if math.isnan(float_number) or math.isinf(float_number):
		return float_number
	multiplier = 10 ** decimal_places
	return int(float_number * multiplier) / multiplier


def plot_results(history, save_path):
	plt.figure(figsize=(20, 10))
	plt.subplot(1, 3, 1)
	plt.ylim((0, numpy.max([i[0] for i in history.losses_distributed])))
	plt.plot([i[0] for i in history.losses_distributed], [i[1] for i in history.losses_distributed])
	plt.xlabel("Round")
	plt.ylabel("Loss")
	plt.grid(True)
	plt.tight_layout()
	plt.subplot(1, 3, 2)
	plt.ylim((0, numpy.max([i[0] for i in history.metrics_distributed['avg_rmse']])))
	plt.plot([i[0] for i in history.metrics_distributed['avg_rmse']],
	         [i[1] for i in history.metrics_distributed['avg_rmse']])
	plt.xlabel("Round")
	plt.ylabel("RMSE")
	plt.grid(True)
	plt.tight_layout()
	plt.subplot(1, 3, 3)
	plt.ylim((0, 1))
	plt.plot([i[0] for i in history.metrics_distributed['avg_pearson_score']],
	         [i[1] for i in history.metrics_distributed['avg_pearson_score']])
	plt.xlabel("Round")
	plt.ylabel("PCC")
	plt.grid(True)
	plt.tight_layout()
	# output, strategy_name, aug, clients, rounds, epochs = plot_params
	plt.savefig(f"{save_path}_distributed.png")
	# plot centralized results
	try:
		plt.figure(figsize=(20, 10))
		plt.subplot(1, 3, 1)
		plt.ylim((0, numpy.max([i[0] for i in history.losses_centralized])))
		plt.plot([i[0] for i in history.losses_centralized], [i[1] for i in history.losses_centralized])
		plt.xlabel("Round")
		plt.ylabel("Loss")
		plt.grid(True)
		plt.tight_layout()
		plt.subplot(1, 3, 2)
		plt.ylim((0, numpy.max([i[0] for i in history.metrics_centralized['avg_rmse']])))
		plt.plot([i[0] for i in history.metrics_centralized['avg_rmse']],
		         [i[1] for i in history.metrics_centralized['avg_rmse']])
		plt.xlabel("Round")
		plt.ylabel("RMSE")
		plt.grid(True)
		plt.tight_layout()
		plt.subplot(1, 3, 3)
		plt.ylim((0, 1))
		plt.plot([i[0] for i in history.metrics_centralized['avg_pearson_score']],
		         [i[1] for i in history.metrics_centralized['avg_pearson_score']])
		plt.xlabel("Round")
		plt.ylabel("PCC")
		plt.grid(True)
		plt.tight_layout()
		# output, strategy_name, aug, clients, rounds, epochs = plot_params
		plt.savefig(f"{save_path}_centralized.png")
	except:
		print("No centralized results")
	
	# save history as json
	with open(f"{save_path}.txt", "w") as f:
		f.write(str(history.__dict__))


def Average(lst):
	return sum(lst) / len(lst)


def get_parameters(net) -> List[np.ndarray]:
	# use val.gpu for gpu
	net.train()
	return [val.cpu().numpy() for _, val in net.state_dict().items()]


def get_parameters_bn(net) -> List[np.ndarray]:
	# Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
	net.train()
	return [val.cpu().numpy() for name, val in net.state_dict().items() if "bn" not in name]


def set_parameters(net, parameters: List[np.ndarray]):
	net.train()
	params_dict = zip(net.state_dict().keys(), parameters)
	state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=True)
	return net


def set_parameters_bn(net, parameters: List[np.ndarray]):
	net.train()
	keys = [k for k in net.state_dict().keys() if "bn" not in k]
	params_dict = zip(keys, parameters)
	state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
	net.load_state_dict(state_dict, strict=False)
	return net


def train(model, train_loader, epochs, DEVICE):
	ramu = RAMU()
	peak_ramu = 0
	peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
	criterion = nn.MSELoss()  # Use appropriate loss function based on your task
	# criterion = nn.L1Loss()  # Use appropriate loss function based on your task
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	model.to(DEVICE)
	model.train()
	
	for epoch in range(epochs):
		running_loss = 0.0
		for images, labels in train_loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			peak_ramu=max(peak_ramu, ramu.compute("TRAINING"))
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
	return peak_ramu


def train_scaffold(model, train_loader, epochs, DEVICE, correction):
	"""Local Adam training with SCAFFOLD's (Karimireddy et al., ICML 2020,
	https://arxiv.org/abs/1910.06378) control-variate gradient correction
	added after backward() and before optimizer.step(). `correction` is
	`global_c - client_c` for this client this round, computed and owned
	server-side (server/strategies.py's SCAFFOLDStrategy -- see its
	docstring for why control-variate state lives there rather than being
	persisted per-client). `correction` is the same length/order as
	model.state_dict(); BatchNorm running_mean/running_var/num_batches_tracked
	positions are always zero there (a buffer has no .grad at all, so a
	correction wouldn't mean anything for it) and are skipped below rather
	than applied. See OFFICEDB_MODIFICATIONS.md item 31.
	"""
	ramu = RAMU()
	peak_ramu = 0
	peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	model.to(DEVICE)
	model.train()

	# Same buffer tags server/strategies.py's bn_buffer_mask matches on --
	# not imported from there to avoid a client/utils -> server import (this
	# module is imported by both sides).
	buffer_tags = ("running_mean", "running_var", "num_batches_tracked")
	is_buffer = [any(tag in k for tag in buffer_tags) for k in model.state_dict().keys()]
	correction_tensors = [
		torch.tensor(c, device=DEVICE, dtype=torch.float32)
		for c, buf in zip(correction, is_buffer) if not buf
	]
	trainable_params = list(model.parameters())
	assert len(correction_tensors) == len(trainable_params), (
		"SCAFFOLD correction/parameter count mismatch: "
		f"{len(correction_tensors)} non-buffer correction entries vs. "
		f"{len(trainable_params)} model.parameters() -- state_dict()'s "
		"non-buffer subsequence is assumed to align with model.parameters() "
		"order (see train_scaffold's docstring); that assumption broke for "
		"this model."
	)

	for epoch in range(epochs):
		running_loss = 0.0
		for images, labels in train_loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			with torch.no_grad():
				for p, corr in zip(trainable_params, correction_tensors):
					if p.grad is not None:
						p.grad.add_(corr)
			optimizer.step()
			running_loss += loss.item()
			peak_ramu = max(peak_ramu, ramu.compute("TRAINING"))
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
	return peak_ramu


def train_with_early_stopping(model, train_loader, val_loader, DEVICE, y_labels, max_epochs=40, patience=5):
	# Val-loss-based early stopping + best-checkpoint selection
	# (OFFICEDB_MODIFICATIONS.md item 25), added because the domain-transfer
	# experiment's original fixed --epochs count had no way to tell
	# "converged" apart from "still improving" or "already overfitting" --
	# per-epoch training-loss logs showed Office-domain checkpoints
	# plateauing by ~epoch 6-8 while Home-domain checkpoints were still
	# declining at epoch 10 for the same fixed budget. Standard practice
	# (e.g. Prechelt 1998, "Early Stopping -- But When?"): track validation
	# loss every epoch, keep a copy of the model's weights whenever a new
	# best is seen, and stop once `patience` epochs pass with no
	# improvement -- the best-seen epoch is usually several epochs before
	# the stopping point itself, not the final epoch trained.
	#
	# Deliberately a new function alongside train(), not a modification of
	# it: train() is also called from run_fl.py/main_fcl.py's federated
	# paths (per-round local training, where "epochs" means local epochs
	# per round, not "train until convergence") -- those call sites are
	# unaffected by this addition.
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	model.to(DEVICE)

	best_val_loss = float('inf')
	best_state = None
	best_epoch = 0
	epochs_without_improvement = 0

	for epoch in range(max_epochs):
		model.train()
		running_loss = 0.0
		for images, labels in train_loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		train_loss = running_loss / len(train_loader)

		val_loss, val_pcc, val_rmse = test(net=model, testloader=val_loader, y_labels=y_labels, DEVICE=DEVICE)
		is_best = val_loss < best_val_loss
		print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}"
			  f"{' (best)' if is_best else ''}")

		if is_best:
			best_val_loss = val_loss
			best_state = copy.deepcopy(model.state_dict())
			best_epoch = epoch + 1
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= patience:
				print(f"Early stopping at epoch {epoch + 1} "
					  f"(best was epoch {best_epoch}, val loss {best_val_loss})")
				break

	model.load_state_dict(best_state)
	return best_epoch, best_val_loss


def pearson_correlation(labels, outputs):
	# Convert batches of lists to numpy arrays for easier computation
	labels_array = np.array(labels)
	outputs_array = np.array(outputs)
	
	# Calculate mean of labels and outputs along the appropriate axis (usually axis=0 for batches)
	mean_labels = np.mean(labels_array, axis=0)
	mean_outputs = np.mean(outputs_array, axis=0)
	
	# Calculate Pearson correlation coefficient along the appropriate axis
	numerator = np.sum((labels_array - mean_labels) * (outputs_array - mean_outputs), axis=0)
	denominator = np.sqrt(np.sum((labels_array - mean_labels) ** 2, axis=0) * np.sum((outputs_array - mean_outputs) ** 2, axis=0))
	
	# Handling division by zero
	zero_denominator_indices = np.where(denominator == 0)
	correlation = np.zeros_like(denominator)
	correlation[zero_denominator_indices] = 0
	non_zero_indices = np.where(denominator != 0)
	correlation[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]
	
	correlation = np.mean(correlation)
	
	return correlation


def test(net, testloader, y_labels, DEVICE, active_idx=None):
	# active_idx: optional list of output-column indices to restrict
	# loss/rmse/pcc to (FCL Axis A action-subset masking -- see
	# OFFICEDB_MODIFICATIONS.md). y_labels must already correspond 1:1 to
	# active_idx (i.e. len(y_labels) == len(active_idx)) when given; None
	# (default) reproduces the original full-output-space behaviour exactly.
	#
	# Metrics are computed once over the FULL concatenated test set, not
	# per-batch-then-averaged (OFFICEDB_MODIFICATIONS.md item 21): the old
	# per-batch Pearson computation went NaN whenever a single batch (most
	# often the final, sub-batch_size batch a non-drop_last test DataLoader
	# produces) happened to have zero-variance labels for one action --
	# common with OfficeDB's small per-client partitions and 1-5 discrete
	# rating scale. The existing NaN-rescue (perturbing outputs with tiny
	# noise) only helps when *predictions* are constant; it can't fix
	# constant *labels*, which is what a degenerate small batch is. The
	# resulting NaN then got averaged in via a non-NaN-safe Average(),
	# silently poisoning that client's ENTIRE avg_pearson_score, every
	# round (the test DataLoader isn't reshuffled between rounds, so it's
	# the same degenerate batch every time). Confirmed empirically: e.g.
	# by-room's Hallway client (n=229 test rows, batch_size=16) has a final
	# 5-sample batch where all 5 rows rate "Carry Drinks"/"Carry Small
	# Objects" identically. Full-partition variance is healthy everywhere
	# (std ~1.2-1.4 on the 1-5 scale) -- computing over the whole held-out
	# set instead of noisy <=16-sample chunks removes the NaN risk
	# entirely, not just this one instance of it.
	criterion = torch.nn.MSELoss()
	RMSE = RMSELoss()
	net.eval()
	net = net.to(DEVICE)
	all_labels, all_outputs = [], []
	with torch.no_grad():
		for features, labels in testloader:
			features, labels = features.to(DEVICE), labels.to(DEVICE)
			outputs = net(features)
			if active_idx is not None:
				outputs = outputs[:, active_idx]
				labels = labels[:, active_idx]
			if torch.isnan(features).any():
				print("-----------------------------features")
			if torch.isnan(labels).any():
				print("-----------------------------labels")
			if torch.isnan(outputs).any():
				print("-----------------------------outputs")

			all_labels.append(labels.cpu())
			all_outputs.append(outputs.cpu())

	all_labels = torch.cat(all_labels, dim=0)
	all_outputs = torch.cat(all_outputs, dim=0)
	loss = criterion(all_outputs, all_labels).item()
	rmse = RMSE(all_outputs, all_labels).item()

	labels_np = all_labels.numpy()
	outputs_np = all_outputs.numpy()
	pearson = {}
	for i in range(len(y_labels)):
		temp = pearsonr(labels_np[:, i], outputs_np[:, i])[0]
		if math.isnan(temp):
			temp = pearsonr(labels_np[:, i], np.random.normal(outputs_np[:, i], 0.0000001))[0]
		pearson[y_labels[i]] = temp
	return loss, Average(list(pearson.values())), rmse


# return loss, pcc, rmse


def predict(net, trainloader, DEVICE, batch_size=16):
	new_pairs = []
	net.eval()  # Set the model to evaluation mode
	net.to(DEVICE)
	with torch.no_grad():
		for inputs, labels in trainloader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			outputs = net(inputs)  # Forward pass through the model
			for i in range(len(outputs)):
				new_pairs.append((outputs[i], labels[i]))
	# batch_size = 1
	new_data = CustomOutputDataset(new_pairs)
	new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
	return new_data_loader


def predict_gen(net, trainloader, DEVICE, batch_size=16):
	new_pairs = []
	net.eval()  # Set the model to evaluation mode
	net.to(DEVICE)
	with torch.no_grad():
		for inputs, labels in trainloader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			outputs = net(inputs)  # Forward pass through the model
			for i in range(len(outputs)):
				new_pairs.append((outputs[i], outputs[i]))
	# batch_size = 1
	new_data = CustomOutputDataset(new_pairs)
	new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
	return new_data_loader


# def predict_gen_distil(net, trainloader):
#     new_pairs = []
#     net.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         for inputs, labels in trainloader:
#             outputs = net(inputs)  # Forward pass through the model
#             new_pairs.append((inputs, outputs))
#     batch_size = 1
#     new_data = CustomOutputDataset(new_pairs) 
#     new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True)
#     return new_data_loader

def predict_gen_distil(net, trainloader, DEVICE, batch_size=16):
	# create new dataframe with predictions of net as labels and images as features keeping in mind that trainloader has batchsize 16
	new_pairs = []
	net.to(DEVICE)
	net.eval()  # Set the model to evaluation mode
	outputs = []
	with torch.no_grad():
		for inputs, labels in trainloader:
			inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
			outputs.append(net(inputs).cpu().numpy())
	# 		for i in range(len(outputs)):
	# 			new_pairs.append((inputs[i], outputs[i]))
	# return DataLoader(CustomOutputDataset(new_pairs), batch_size=batch_size, shuffle=True)
	num_classes = outputs[0].shape[-1]
	return numpy.asarray(outputs).reshape((len(trainloader) * batch_size, num_classes))


def get_eval_fn(net, testloader, y_labels, DEVICE, checkpoint_path=None):
	# checkpoint_path: optional, dumps the aggregated global model each round
	# via _dump_global_params -- same pattern as get_eval_fn_cl's checkpoint_path
	# (added for Track 2 federated domain-transfer; see OFFICEDB_MODIFICATIONS.md).
	# None (default) reproduces original behavior exactly -- no callers besides
	# main.py's FedAvg branch pass this today.
	def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
		# print(config.keys())
		net.train()
		params_dict = zip(net.state_dict().keys(), weights)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		net.load_state_dict(state_dict, strict=True)
		if checkpoint_path is not None:
			_dump_global_params(net, checkpoint_path)
		loss, avg_pearson, avg_rmse = test(net, testloader, y_labels, DEVICE)
		print("Round %s, Loss %s, Pearson %s, RMSE %s" % (server_round, loss, avg_pearson, avg_rmse))
		return loss, {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}

	return evaluate


def _dump_global_params(reference_module, path):
	# Chained-job resume (OFFICEDB_MODIFICATIONS.md item 25): pickles a plain
	# state_dict, same convention as pretrain.py's pickle.dump(model.state_dict(), f).
	# Written atomically (tmp file + os.replace) so a job killed mid-write
	# (e.g. hitting the Slurm wall-time cap) can't leave a half-written,
	# unloadable checkpoint for the next chained job to trip over.
	tmp = f"{path}.tmp"
	with open(tmp, 'wb') as f:
		pickle.dump(reference_module.state_dict(), f)
	os.replace(tmp, path)


def make_checkpoint_hook(reference_module, checkpoint_path):
	# Chained-job resume (OFFICEDB_MODIFICATIONS.md item 25): the FedRoot
	# strategy branches in main_fcl.py construct FedAvgWithAccuracyMetric with
	# no evaluate_fn at all (no centralized eval today), so there's no
	# existing per-round hook there to hang a checkpoint dump off of. This is
	# a checkpoint-only stand-in -- always returns None, which flwr 1.12.0's
	# Strategy.evaluate() treats as "no centralized eval this round" (see
	# flwr/server/strategy/fedavg.py), leaving history.losses_centralized
	# untouched exactly as today's no-evaluate_fn behavior.
	def evaluate(server_round, weights, config):
		params_dict = zip(reference_module.state_dict().keys(), weights)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		reference_module.load_state_dict(state_dict, strict=True)
		_dump_global_params(reference_module, checkpoint_path)
		return None
	return evaluate


def get_eval_fn_cl(net, testloader, y_labels, DEVICE, rounds_per_task=5, n_tasks=2, active_idx_per_task=None, round_offset=0, checkpoint_path=None):
	# testloader is the cumulative per-task-boundary test loader list from
	# dataloader.utils.task_splitter (tasks 0..task_idx combined) -- see
	# OFFICEDB_MODIFICATIONS.md. Originally hardcoded to "<=5"/testloader[0]
	# vs. testloader[1], and averaged two separately-computed test() results
	# (mean of per-task means) rather than evaluating once over the pooled
	# cumulative set (sample-weighted); for n_tasks=2 both still land on the
	# same two loaders, just computed as one pooled test() call instead of
	# two averaged ones -- an intentional, more standard combined-eval
	# computation, not a behavior bug.
	#
	# active_idx_per_task: optional list (len n_tasks) of cumulative active
	# output-column indices for FCL Axis A (action-subset masking) -- None
	# (default) reproduces the original unmasked full-output-space behaviour.
	def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
		net.train()
		params_dict = zip(net.state_dict().keys(), weights)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		net.load_state_dict(state_dict, strict=True)
		if checkpoint_path is not None:
			_dump_global_params(net, checkpoint_path)
		effective_round = server_round + round_offset  # OFFICEDB_MODIFICATIONS.md item 25
		task_idx = min((effective_round - 1) // rounds_per_task, n_tasks - 1)
		if active_idx_per_task is not None:
			active_idx = active_idx_per_task[task_idx]
			task_y_labels = [y_labels[i] for i in active_idx]
		else:
			active_idx = None
			task_y_labels = y_labels
		loss, avg_pearson, avg_rmse = test(net, testloader[task_idx], task_y_labels, DEVICE, active_idx=active_idx)

		print("Round %s, Loss %s, Pearson %s, RMSE %s" % (server_round, loss, avg_pearson, avg_rmse))
		return loss, {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}

	return evaluate


def get_eval_fn_bn(net, testloader, y_labels, DEVICE):
	def evaluate(server_round: int, weights: fl.common.NDArrays, config: Dict[str, Scalar]) -> Optional[
		Tuple[float, Dict[str, Scalar]]]:
		net.train()
		keys = [k for k in net.state_dict().keys() if "bn" not in k]
		params_dict = zip(keys, weights)
		
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		net.load_state_dict(state_dict, strict=False)
		
		loss, avg_pearson, avg_rmse = test(net, testloader, y_labels, DEVICE)
		return loss, {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
	
	return evaluate


def extract_metrics_gpu_csv(filename):
	df = pd.read_csv(filename)
	# create a new dataframe with RAM, CPU, GPU usage
	df2 = pd.DataFrame(columns=["Strategy", "RAM", "CPU", "GPU_U", "GPU_Mem"])
	for index, row in df.iterrows():
		ram = float(row["RAM After"]) - float(row["RAM Before"])
		ram *= 1024
		cpu_af = list(map(float, row["CPU After"].replace('(', '').replace(')', '').split(', ')))
		cpu_bef = list(map(float, row["CPU Before"].replace('(', '').replace(')', '').split(', ')))
		cpu = cpu_af[0] - cpu_bef[0] + cpu_af[1] - cpu_bef[1]
		# cpu=float(row["CPU After"][0])-float(row["CPU Before"][0]) + float(row["CPU After"][1])-float(row["CPU Before"][1])
		try:
			gpu_af = list(map(float, row["GPU After"].replace('(', '').replace(')', '').split(', ')))
			gpu_bef = list(map(float, row["GPU Before"].replace('(', '').replace(')', '').split(', ')))
			gpu_u = gpu_af[0] - gpu_bef[0]
			gpu_mem = gpu_af[1] - gpu_bef[1]
		except:
			gpu_u = 0
			gpu_mem = 0
		# gpu_u=float(row["GPU After"][0])-float(row["GPU Before"][0])
		# gpu_mem=float(row["GPU After"][1])-float(row["GPU Before"][1])
		df2.loc[index] = [row["Strategy"], ram, cpu, gpu_u, gpu_mem]
	return df2
