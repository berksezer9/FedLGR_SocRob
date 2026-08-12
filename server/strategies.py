from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import (
	EvaluateRes,
	FitRes,
	Parameters,
	Scalar,
	ndarrays_to_parameters,
	parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


def bn_buffer_mask(state_dict_keys) -> List[bool]:
	"""True at index i if the i-th entry of an ordered state_dict is a
	BatchNorm running_mean/running_var/num_batches_tracked buffer rather than
	a trainable weight/bias -- see FedOptAdamStrategy and
	OFFICEDB_MODIFICATIONS.md item 19. Order must match whatever
	get_parameters(...)/net.state_dict() order was used to build the
	corresponding initial_parameters."""
	return [any(tag in k for tag in ("running_mean", "running_var", "num_batches_tracked"))
	        for k in state_dict_keys]

class RoundFailedError(RuntimeError):
	"""Raised when every client failed/timed out in a round (OFFICEDB_MODIFICATIONS.md
	item 22): a hard stop instead of Flower's default (silently return None/{}
	and let the simulation carry on to the next round with a no-op). Added
	after observing that a hung round (see FL_ROUND_TIMEOUT, main.py/
	main_fcl.py) doesn't recover on its own -- the same 0-results outcome
	repeated on the very next round too -- so letting the simulation continue
	just burns the rest of the job's wall-time logging meaningless rounds
	instead of failing fast and visibly (non-zero exit code, easy to detect
	via sacct/grep and retry) the first time it happens."""


def _require_results(results, failures, phase: str, server_round: int) -> None:
	if not results:
		raise RoundFailedError(
			f"{phase} round {server_round}: 0/{len(results) + len(failures)} clients "
			f"returned a result (all {len(failures)} failed/timed out) -- aborting instead "
			f"of continuing with a no-op round. See OFFICEDB_MODIFICATIONS.md item 22."
		)


def weighted_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
	"""Aggregate evaluation results obtained from multiple clients."""
	num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
	weighted_values = [num_examples * value for num_examples, value in results]
	
	return sum(weighted_values) / num_total_evaluation_examples


class FedAvgWithAccuracyMetric(fl.server.strategy.FedAvg):
	def aggregate_fit(self, server_round, results, failures):
		_require_results(results, failures, "fit", server_round)
		return super().aggregate_fit(server_round, results, failures)

	def aggregate_evaluate(self,
						   rnd: int,
						   results: List[Tuple[ClientProxy, EvaluateRes]],
						   failures: List[BaseException],
						   ) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""

		_require_results(results, failures, "evaluate", rnd)
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		pcc_aggregated = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_pearson_score']) for _, evaluate_res in results])
		rmse = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_rmse']) for _, evaluate_res in results])
		
		return loss_aggregated, {'avg_pearson_score': pcc_aggregated, 'avg_rmse': rmse}


class FedProxWithAccuracyMetric(fl.server.strategy.FedProx):
	def aggregate_fit(self, server_round, results, failures):
		_require_results(results, failures, "fit", server_round)
		return super().aggregate_fit(server_round, results, failures)

	def aggregate_evaluate(
		self,
		rnd: int,
		results: List[Tuple[ClientProxy, EvaluateRes]],
		failures: List[BaseException],
	) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		_require_results(results, failures, "evaluate", rnd)
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		pcc_aggregated = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_pearson_score']) for _, evaluate_res in results])
		rmse = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_rmse']) for _, evaluate_res in results])
		
		return loss_aggregated, {'avg_pearson_score': pcc_aggregated, 'avg_rmse':rmse}


class FedNovaStrategy(fl.server.strategy.FedAvg):
	"""FedNova (Wang et al., NeurIPS 2020, https://arxiv.org/abs/2007.07481):
	normalizes each client's update by how many local steps it actually took
	before averaging, correcting FedAvg's implicit bias toward clients that
	run more local computation (objective inconsistency under heterogeneous
	local step counts). See OFFICEDB_MODIFICATIONS.md item 28.

	Per-client normalized update: d_i = (x^t - y_i) / tau_i, where x^t is the
	global model this round started from, y_i is client i's returned weights,
	and tau_i is client i's local step count (`local_steps` in its fit()
	metrics -- client/default.py's FlowerClient.fit()). Server step:
	x^{t+1} = x^t - tau_eff * sum_i(p_i * d_i), with p_i = n_i / sum(n_j)
	(n_i = FitRes.num_examples, same weighting FedAvg already uses) and
	tau_eff = sum_i(p_i * tau_i).

	Unlike FedAvg's plain weighted average (a convex combination, so always
	bounded within the clients' value range), this is a genuine
	extrapolation -- sum_i(p_i * tau_eff / tau_i) need not equal 1, so the
	result can overshoot past x^t/y_i's range. Applied to BatchNorm
	running_var, that overshoot can push it negative -> NaN in the very next
	forward pass, the exact failure mode item 19 already found for
	FedOptAdamStrategy's unmasked FedAdam. Reuses that item's fix: buffers
	(identified via `buffer_mask`, e.g. `bn_buffer_mask(...)`) are instead
	plain weighted-averaged; `buffer_mask=None` (default) applies the
	FedNova extrapolation everywhere, matching FedOptAdamStrategy's own
	`buffer_mask=None` default semantics.
	"""

	def __init__(self, *args, buffer_mask: Optional[List[bool]] = None, **kwargs):
		if kwargs.get("initial_parameters") is None:
			raise ValueError(
				"FedNovaStrategy requires initial_parameters -- it needs the "
				"pre-round global weights (self.current_weights) to compute each "
				"client's normalized delta, unlike plain FedAvg which only "
				"averages the clients' returned weights directly."
			)
		super().__init__(*args, **kwargs)
		self.current_weights = parameters_to_ndarrays(kwargs["initial_parameters"])
		self.buffer_mask = buffer_mask

	def aggregate_fit(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, FitRes]],
		failures: List[Any],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		_require_results(results, failures, "fit", server_round)
		if not self.accept_failures and failures:
			return None, {}

		total_examples = sum(fit_res.num_examples for _, fit_res in results)
		tau_effective = 0.0
		# (p_i, delta_i, client_weights) per client -- client_weights kept
		# around (not just delta_i) so the buffer branch below can plain-
		# average the clients' raw returned values without re-deserializing
		# fit_res.parameters a second time per buffer index.
		per_client: List[Tuple[float, List[np.ndarray], List[np.ndarray]]] = []
		for _, fit_res in results:
			tau_i = fit_res.metrics.get("local_steps")
			if not tau_i:
				raise RoundFailedError(
					f"fit round {server_round}: a client returned no (or zero) "
					"'local_steps' fit-metric -- FedNovaStrategy requires it "
					"(client/default.py's FlowerClient.fit() sets it; using a "
					"different client class with FedNova is unsupported). See "
					"OFFICEDB_MODIFICATIONS.md item 28."
				)
			p_i = fit_res.num_examples / total_examples
			tau_effective += p_i * tau_i
			client_weights = parameters_to_ndarrays(fit_res.parameters)
			delta_i = [
				(x - y) / tau_i for x, y in zip(self.current_weights, client_weights)
			]
			per_client.append((p_i, delta_i, client_weights))

		aggregated_delta = [np.zeros_like(x) for x in self.current_weights]
		for p_i, delta_i, _ in per_client:
			for j, d in enumerate(delta_i):
				aggregated_delta[j] = aggregated_delta[j] + p_i * d

		new_weights = []
		for j, (x, agg) in enumerate(zip(self.current_weights, aggregated_delta)):
			if self.buffer_mask and self.buffer_mask[j]:
				# Plain weighted average of the clients' returned buffer values
				# instead of the FedNova extrapolation -- see class docstring /
				# item 28 (mirrors item 19's FedOptAdamStrategy fix).
				new_weights.append(
					sum(p_i * client_weights[j] for p_i, _, client_weights in per_client)
				)
			else:
				new_weights.append(x - tau_effective * agg)

		self.current_weights = new_weights

		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		return ndarrays_to_parameters(new_weights), metrics_aggregated

	def aggregate_evaluate(
		self,
		rnd: int,
		results: List[Tuple[ClientProxy, EvaluateRes]],
		failures: List[BaseException],
	) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		_require_results(results, failures, "evaluate", rnd)
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		pcc_aggregated = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_pearson_score']) for _, evaluate_res in results])
		rmse = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_rmse']) for _, evaluate_res in results])

		return loss_aggregated, {'avg_pearson_score': pcc_aggregated, 'avg_rmse': rmse}


class FedOptAdamStrategy(fl.server.strategy.FedAdam):
	# Was fl.server.strategy.FedAvg with a hand-rolled `aggregate(self, reports)`
	# override -- Flower's Strategy interface calls `aggregate_fit(self,
	# server_round, results, failures)`, not `aggregate`, so that method was
	# dead code and this class silently ran plain FedAvg aggregation under the
	# FedOptAdam label (see OFFICEDB_MODIFICATIONS.md item 17). Now subclasses
	# flwr's own FedAdam (Reddi et al. 2020) for real Adam-style aggregation --
	# but restricted to actual trainable parameters via `aggregate_fit` below,
	# not applied wholesale (see that method's docstring / item 19).
	def __init__(self, *args, buffer_mask: Optional[List[bool]] = None, **kwargs):
		super().__init__(*args, **kwargs)
		# buffer_mask[i] == True: index i of the flat parameter list is a
		# BatchNorm running_mean/running_var/num_batches_tracked buffer, not a
		# trainable weight -- see bn_buffer_mask() above. None (default)
		# reproduces flwr's plain FedAdam behaviour (Adam applied to every
		# entry) exactly.
		self.buffer_mask = buffer_mask

	def aggregate_fit(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, FitRes]],
		failures: List[Any],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		"""Adam-normalize only the trainable-parameter entries; BatchNorm's
		running_mean/running_var/num_batches_tracked are plain weighted-averaged
		instead (flwr's FedAvg.aggregate_fit, called directly below, bypassing
		FedAdam/FedOpt in the MRO).

		Applying flwr's stock FedAdam to a raw net.state_dict() (its default
		usage, and what this class did before this fix) treats every entry --
		including running_var, which must stay non-negative -- as a free
		Adam-optimized parameter. Adam's per-parameter update magnitude is
		normalized to ~eta regardless of eta's size (that's the point of the
		normalization), so for a BatchNorm channel whose running_var is smaller
		than eta, the "even a small eta is safe" assumption doesn't hold: it just
		shrinks how many channels get pushed negative, not whether any do.
		Confirmed empirically against a fresh MobileNet state_dict: eta=0.1 pushed
		49/53 running_var tensors negative, eta=0.0001 still pushed 12/53 negative.
		A negative running_var hits `sqrt(var + eps)` in the very next BatchNorm
		forward pass and produces NaN outputs network-wide -- exactly what the
		2026-07-31 by-robot/FedOptAdam smoke test hit at round 1 (finite at round
		0, NaN loss/RMSE/PCC from round 1 on). See OFFICEDB_MODIFICATIONS.md item
		19. `buffer_mask=None` (default) reproduces flwr's plain FedAdam exactly.
		"""
		_require_results(results, failures, "fit", server_round)
		if not self.buffer_mask:
			return super().aggregate_fit(server_round, results, failures)

		fedavg_parameters_aggregated, metrics_aggregated = fl.server.strategy.FedAvg.aggregate_fit(
			self, server_round, results, failures
		)
		if fedavg_parameters_aggregated is None:
			return None, {}
		fedavg_weights = parameters_to_ndarrays(fedavg_parameters_aggregated)

		delta_t = [x - y for x, y in zip(fedavg_weights, self.current_weights)]
		if not self.m_t:
			self.m_t = [np.zeros_like(x) for x in delta_t]
		if not self.v_t:
			self.v_t = [np.zeros_like(x) for x in delta_t]

		new_weights = []
		for i, is_buffer in enumerate(self.buffer_mask):
			if is_buffer:
				# Plain weighted average (already non-negative-preserving,
				# unlike an Adam-normalized step) -- m_t/v_t left untouched
				# (unused for this index going forward).
				new_weights.append(fedavg_weights[i])
				continue
			self.m_t[i] = self.beta_1 * self.m_t[i] + (1 - self.beta_1) * delta_t[i]
			self.v_t[i] = self.beta_2 * self.v_t[i] + (1 - self.beta_2) * (delta_t[i] * delta_t[i])
			new_weights.append(
				self.current_weights[i] + self.eta * self.m_t[i] / (np.sqrt(self.v_t[i]) + self.tau)
			)

		self.current_weights = new_weights
		return ndarrays_to_parameters(self.current_weights), metrics_aggregated

	def aggregate_evaluate(
		self,
		rnd: int,
		results: List[Tuple[ClientProxy, EvaluateRes]],
		failures: List[BaseException],
	) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		_require_results(results, failures, "evaluate", rnd)
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		pcc_aggregated = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_pearson_score']) for _, evaluate_res in results])
		rmse = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_rmse']) for _, evaluate_res in results])

		return loss_aggregated, {'avg_pearson_score': pcc_aggregated, 'avg_rmse':rmse}
