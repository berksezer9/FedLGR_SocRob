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

def weighted_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
	"""Aggregate evaluation results obtained from multiple clients."""
	num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
	weighted_values = [num_examples * value for num_examples, value in results]
	
	return sum(weighted_values) / num_total_evaluation_examples


class FedAvgWithAccuracyMetric(fl.server.strategy.FedAvg):
	def aggregate_evaluate(self,
						   rnd: int,
						   results: List[Tuple[ClientProxy, EvaluateRes]],
						   failures: List[BaseException],
						   ) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		
		if not results:
			return None, {}
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
	def aggregate_evaluate(
		self,
		rnd: int,
		results: List[Tuple[ClientProxy, EvaluateRes]],
		failures: List[BaseException],
	) -> Tuple[Optional[float], Dict[str, Scalar]]:
		"""Aggregate evaluation losses using weighted average."""
		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		pcc_aggregated = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_pearson_score']) for _, evaluate_res in results])
		rmse = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_rmse']) for _, evaluate_res in results])
		
		return loss_aggregated, {'avg_pearson_score': pcc_aggregated, 'avg_rmse':rmse}


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
		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}
		loss_aggregated = weighted_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
		pcc_aggregated = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_pearson_score']) for _, evaluate_res in results])
		rmse = weighted_avg(
			[(evaluate_res.num_examples, evaluate_res.metrics['avg_rmse']) for _, evaluate_res in results])
		
		return loss_aggregated, {'avg_pearson_score': pcc_aggregated, 'avg_rmse':rmse}
