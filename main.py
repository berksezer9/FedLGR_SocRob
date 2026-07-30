import argparse
import ray
import flwr as fl
import torch.nn as nn
from datetime import datetime

from server.strategies import FedAvgWithAccuracyMetric, FedProxWithAccuracyMetric, FedOptAdamStrategy
from server.utils import fit_config, evaluate_config

from client.fedBN import FlowerClient_BN, FlowerClient_BN_Root
from client.default import FlowerClient
from client.fedRoot import FlowerClient_Root
from dataloader.utils import *

from utils import plot_results, get_parameters, set_parameters, train, test, predict, predict_gen, get_parameters_bn, \
    get_eval_fn, get_eval_fn_bn, extract_metrics_gpu_csv, truncate_float

from metrics.computation import GPUUsage, RAMU, CPUUsage
from distutils.util import strtobool
import torch
torch.manual_seed(1024)
import numpy as np
import pandas as pd
np.random.seed(1024)
import psutil
import gc
from typing import Dict, List, Optional, Tuple
from flwr.common import NDArrays, Scalar


def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


def run_strategy(strategy, strategy_name, client_function, clients, rounds, epochs, output, aug, ray_init_args, client_res):
    """
	   Running all strategies defined under "strategy"
	"""
    print("Running strategy " + str(strategy_name) + " for " + str(clients) + "clients!")
    history = fl.simulation.start_simulation(
        client_fn=client_function,
        num_clients=clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        client_resources=client_res)

    try:
        data = pd.read_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_decentral.csv")
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        data = pd.DataFrame(columns=["Method", "Loss", "RMSE", "PCC"])

    data = pd.concat([data, pd.Series([f"{strategy_name}", truncate_float(history.losses_distributed[-1][-1], 4),
                                       truncate_float(history.metrics_distributed['avg_rmse'][-1][-1],4),
                                       truncate_float(history.metrics_distributed['avg_pearson_score'][-1][-1],4)],
                                      index=data.columns).to_frame().T])
    data.to_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_decentral.csv")

    try:
        try:
            data = pd.read_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_central.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data = pd.DataFrame(columns=["Method", "Loss", "RMSE", "PCC"])
        
        #use truncate_float to truncate the float values to 4 decimal places
        data= pd.concat([data, pd.Series([f"{strategy_name}", truncate_float(history.losses_centralized[-1][-1], 4),
                                             truncate_float(history.metrics_centralized['avg_rmse'][-1][-1],4),
                                             truncate_float(history.metrics_centralized['avg_pearson_score'][-1][-1],4)],
                                            index=data.columns).to_frame().T])

        # data = pd.concat([data, pd.Series([f"{strategy_name}", history.losses_centralized[-1][-1],
        #                                    history.metrics_centralized['avg_rmse'][-1][-1],
        #                                    history.metrics_centralized['avg_pearson_score'][-1][-1]],
        #                                   index=data.columns).to_frame().T])
        # data.to_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_central.csv")
    except:
        print("No centralized results")

    # plot_params = (output, strategy_name, aug, clients, rounds, epochs)
    if '-' in strategy_name:
        strategy_name, base_name = strategy_name.split('-')
        save_path = f"{output}/{strategy_name}/{strategy_name}-{base_name}_{aug}_{clients}_{rounds}_{epochs}.png"
    else:
        save_path = f"{output}/{strategy_name}/{strategy_name}_{aug}_{clients}_{rounds}_{epochs}.png"

    plot_results(history=history, save_path=save_path)
    del history
    ray.shutdown()
    gc.collect()


def savecomp(output, strat, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf):
    """
		Saving result comparisons for different hyperparameters.
	"""
    try:
        data = pd.read_csv(f"{output}/comp.csv")
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        data = pd.DataFrame(
            columns=["Strategy", "RAM Before", "RAM After", "CPU Before", "CPU After", "GPU Before", "GPU After"])
    data = pd.concat(
        [data, pd.Series([f"{strat}", rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf], index=data.columns).to_frame().T])
    data.to_csv(f"{output}/comp.csv")


def run(args):
    def client_fn(cid):
        return FlowerClient(cid, net.to(DEVICE), trainloaders[int(cid)], testloader, epochs=int(args.epochs),
                            y_labels=y_labels, num_clients=int(n_cl), DEVICE=DEVICE, path=path)

    def client_fn_root(cid):
        return FlowerClient_Root(cid, net.to(DEVICE), trainloaders[int(cid)], testloader, epochs=int(args.epochs),
                                 y_labels=y_labels, num_clients=int(n_cl), DEVICE=DEVICE, path=path)

    def client_fn_BN_Root(cid):
        return FlowerClient_BN_Root(cid, net.to(DEVICE), trainloaders[int(cid)], testloader, epochs=int(args.epochs),
                                    y_labels=y_labels, num_clients=int(n_cl), DEVICE=DEVICE, path=path)

    def client_fn_BN(cid):
        return FlowerClient_BN(cid, net.to(DEVICE), trainloaders[int(cid)], testloader, epochs=int(args.epochs),
                               y_labels=y_labels, num_clients=int(n_cl), DEVICE=DEVICE, path=path)

    if args.model == 'MobileNet':
        from models.MobileNet import Net
    elif args.model == 'DeepLabMobileNet':
        from models.deepLabMobileNet import Net

    # None (default) reproduces vendor's original MANNERS-DB column names; a
    # comma-separated --action_cols/--extra_cols CLI value overrides them
    # (OfficeDB's 9 actions, no circle/arrow columns -- see
    # OFFICEDB_MODIFICATIONS.md).
    action_cols = args.action_cols.split(',') if args.action_cols else None
    extra_cols = args.extra_cols.split(',') if args.extra_cols else None

    experiment_path = f"{args.output}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.strategy}_{args.model}_{args.rounds}_{args.icl}_{args.fcl}_{args.aug}_{args.base}_{args.processor_type}"
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    # update the output path
    args.output = experiment_path
    num_CPUs = args.num_cpus
    if args.processor_type == 'gpu':
        num_GPUs = 1
        ray_init_args = {"num_gpus": num_GPUs, "num_cpus": num_CPUs}
        DEVICE = torch.device("cuda")
        gpu_flag = 1
    else:
        num_GPUs = 0
        ray_init_args = {"num_gpus": num_GPUs, "num_cpus": num_CPUs}
        DEVICE = torch.device("cpu")
        gpu_flag = 0
    # Ray defaults its temp dir to /tmp/ray -- on a shared Wilkes3 GPU node
    # /tmp is node-local but can already be owned by another user's leftover
    # Ray session, surfacing as a PermissionError on
    # /tmp/ray/ray_current_cluster. RAY_TMPDIR (set by the calling sbatch
    # script/smoke test, not a Ray-native env var) lets that be overridden
    # without hardcoding a path here.
    if os.environ.get("RAY_TMPDIR"):
        ray_init_args["_temp_dir"] = os.environ["RAY_TMPDIR"]

    # Initial RAM and CPU Usage.
    ramu = RAMU()
    cpuu = CPUUsage()

    # Initial GPU usage for GPUID
    if gpu_flag == 1:
        gpuu = GPUUsage(0)
    data_permutation = None
    # Running the loop for clients between icl (low) and fcl (high)
    for n_cl in range(int(args.icl), int(args.fcl) + 1):

        # Ray only runs clients concurrently up to floor(pool_cpus / cpus_per_client);
        # dividing by n_cl here (mirroring the num_gpus division just above) is what
        # lets all n_cl clients in a round train in parallel instead of serially --
        # previously every client requested the *entire* CPU pool, capping concurrency
        # at 1 regardless of how many cores the VM actually had. See
        # OFFICEDB_MODIFICATIONS.md.
        if gpu_flag == 1:
            client_res = {"num_gpus": num_GPUs / n_cl, "num_cpus": num_CPUs / n_cl}
        else:
            client_res = {"num_gpus": num_GPUs, "num_cpus": num_CPUs / n_cl}

        if not os.path.exists(f"{args.output}/{n_cl}"):
            os.mkdir(f"{args.output}/{n_cl}")

        # if strategy is all, run all strategies, else run the specified strategy
        if args.strategy == 'all':
            strategies = ['FedAvg', 'FedBN', 'FedOptAdam', 'FedProx', 'FedDistill', 'FedRoot']
        else:
            strategies = [args.strategy]

        for strat in strategies:
            net = Net(num_classes=args.num_classes)
            central_model = Net(num_classes=args.num_classes)

            path = f"{args.output}/{n_cl}/{strat}"
            if not os.path.exists(path):
                os.mkdir(path)
            output = f"{args.output}/{n_cl}"
            # if 'FedBN' in strat:
            #     bn = True
            # else:
            #     bn = False
            if 'Distill' in strat:
                trainloaders, testloader, y_labels, data_permutation = load_datasets(num_clients=int(n_cl), path=args.path, aug=args.aug, batch_size=args.batch_size,
                                                                                     distil=True, out=path, DEVICE=DEVICE,
                                                                                     data_permutation=data_permutation, teacher_model=net,
                                                                                     action_cols=action_cols, extra_cols=extra_cols,
                                                                                     split_col=args.split_col, group_col=args.group_col)
            else:
                trainloaders, testloader, y_labels, data_permutation = load_datasets(num_clients=int(n_cl), path=args.path, aug=args.aug, batch_size=args.batch_size,
                                                                                     DEVICE=DEVICE, data_permutation=data_permutation,
                                                                                     action_cols=action_cols, extra_cols=extra_cols,
                                                                                     split_col=args.split_col, group_col=args.group_col)

            # params = get_parameters(net)
            if strat == 'FedAvg':
                strategy = FedAvgWithAccuracyMetric(
                    min_available_clients=int(n_cl),
                    # initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                    on_evaluate_config_fn=evaluate_config,
                    evaluate_fn=get_eval_fn(central_model, testloader=testloader, DEVICE=DEVICE, y_labels=y_labels)
                )
                client_function = client_fn
            elif strat == 'FedProx':
                strategy = FedProxWithAccuracyMetric(
                    min_available_clients=int(n_cl),
                    # initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                    on_evaluate_config_fn=evaluate_config,
                    proximal_mu=0.1,
                    evaluate_fn=get_eval_fn(central_model, testloader=testloader, DEVICE=DEVICE, y_labels=y_labels)
                )
                client_function = client_fn

            elif strat == 'FedOptAdam':
                strategy = FedOptAdamStrategy(
                    min_available_clients=int(n_cl),
                    # initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                    on_evaluate_config_fn=evaluate_config,
                    evaluate_fn=get_eval_fn(central_model, testloader=testloader, DEVICE=DEVICE, y_labels=y_labels)
                )
                client_function = client_fn

            elif strat == 'FedBN':
                # params = get_parameters_bn(net)
                strategy = FedAvgWithAccuracyMetric(
                    min_available_clients=int(n_cl),
                    # initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                    on_evaluate_config_fn=evaluate_config,
                    evaluate_fn=get_eval_fn_bn(central_model, testloader=testloader, DEVICE=DEVICE, y_labels=y_labels)
                )
                client_function = client_fn_BN
            elif strat == 'FedDistill':
                strategy = FedAvgWithAccuracyMetric(
                    min_available_clients=int(n_cl),
                    # initial_parameters=fl.common.ndarrays_to_parameters(params),
                    on_fit_config_fn=fit_config,
                    on_evaluate_config_fn=evaluate_config,
                    evaluate_fn=get_eval_fn(central_model, testloader=testloader, DEVICE=DEVICE, y_labels=y_labels)
                )
                client_function = client_fn
            elif strat == 'FedRoot':
                root = True
                if args.base == 'all':
                    strategies_base = ['FedAvg', 'FedBN', 'FedOptAdam', 'FedProx', 'FedDistill']
                else:
                    strategies_base = [args.base]
                for base in strategies_base:
                    path = f"{args.output}/{n_cl}/{strat}/{strat}-{base}"

                    # strat = strat + "-" + base
                    if not os.path.exists(path):
                        os.mkdir(path)
                    # if 'FedBN' in base:
                    #     bn = True
                    #     client_function = client_fn_BN_Root
                    # else:
                    #     bn = False
                    #     client_function = client_fn_root

                    if 'Distill' in base:
                        trainloaders, testloader, y_labels, data_permutation = load_datasets(num_clients=int(n_cl), path=args.path, aug=args.aug, batch_size=args.batch_size,
                                                                                            distil=True, out=path, DEVICE=DEVICE,
                                                                                            data_permutation=data_permutation, teacher_model=net,
                                                                                            action_cols=action_cols, extra_cols=extra_cols,
                                                                                            split_col=args.split_col, group_col=args.group_col)
                    else:
                        trainloaders, testloader, y_labels, data_permutation = load_datasets(num_clients=int(n_cl), path=args.path, aug=args.aug, batch_size=args.batch_size,
                                                                                            DEVICE=DEVICE, data_permutation=data_permutation,
                                                                                            action_cols=action_cols, extra_cols=extra_cols,
                                                                                            split_col=args.split_col, group_col=args.group_col)

                    # params = get_parameters(net.conv_module)
                    if base == 'FedAvg':
                        strategy = FedAvgWithAccuracyMetric(
                            min_available_clients=int(n_cl),
                            # initial_parameters=fl.common.ndarrays_to_parameters(params),
                            on_fit_config_fn=fit_config,
                            on_evaluate_config_fn=evaluate_config
                        )
                        client_function = client_fn_root

                    elif base == 'FedProx':
                        strategy = FedProxWithAccuracyMetric(
                            min_available_clients=int(n_cl),
                            # initial_parameters=fl.common.ndarrays_to_parameters(params),
                            on_fit_config_fn=fit_config,
                            on_evaluate_config_fn=evaluate_config,
                            proximal_mu=0.1
                        )
                        client_function = client_fn_root
                    elif base == 'FedOptAdam':
                        strategy = FedOptAdamStrategy(
                            min_available_clients=int(n_cl),
                            # initial_parameters=fl.common.ndarrays_to_parameters(params),
                            on_fit_config_fn=fit_config,
                            on_evaluate_config_fn=evaluate_config
                        )
                        client_function = client_fn_root
                    elif base == 'FedBN':
                        # params = get_parameters_bn(net.conv_module)
                        strategy = FedAvgWithAccuracyMetric(
                            min_available_clients=int(n_cl),
                            # initial_parameters=fl.common.ndarrays_to_parameters(params),
                            on_fit_config_fn=fit_config,
                            on_evaluate_config_fn=evaluate_config
                        )
                        client_function = client_fn_BN_Root

                    elif base == 'FedDistill':
                        strategy = FedAvgWithAccuracyMetric(
                            min_available_clients=int(n_cl),
                            # initial_parameters=fl.common.ndarrays_to_parameters(params),
                            on_fit_config_fn=fit_config,
                            on_evaluate_config_fn=evaluate_config
                        )
                        client_function = client_fn_root
                    else:
                        client_function = None
                        strategy = None
                        raise Exception("Base Strategy ill-defined for FedRoot")

                    rambef = ramu.compute("BEFORE EVALUATION")
                    cpubef = cpuu.compute("BEFORE EVALUATION")

                    if gpu_flag:
                        gpubeff = gpuu.compute("BEFORE EVALUATION")
                    else:
                        gpubeff = 0

                    strategy_name = strat + "-" + base
                    run_strategy(strategy=strategy, strategy_name=strategy_name, client_function=client_function,
                                 clients=n_cl, rounds=args.rounds,
                                 epochs=args.epochs, output=output, aug=args.aug, ray_init_args=ray_init_args,
                                 client_res=client_res)

                    ramaf = ramu.compute("AFTER EVALUATION")
                    cpuaf = cpuu.compute("AFTER EVALUATION")
                    if gpu_flag:
                        gpuaf = gpuu.compute("AFTER EVALUATION")
                    else:
                        gpuaf = 0

                    savecomp(output, strategy_name, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
                    del strategy
                    auto_garbage_collect()
            else:
                client_function = None
                strategy = None
                raise Exception("Strategy ill-defined.")

            rambef = ramu.compute("BEFORE EVALUATION")
            cpubef = cpuu.compute("BEFORE EVALUATION")

            if gpu_flag:
                gpubeff = gpuu.compute("BEFORE EVALUATION")
            else:
                gpubeff = 0

            if strat == "FedRoot":
                continue
            else:
                strategy_name = strat
            run_strategy(strategy=strategy, strategy_name=strategy_name, client_function=client_function, clients=n_cl,
                         rounds=args.rounds,
                         epochs=args.epochs, output=output, aug=args.aug, ray_init_args=ray_init_args,
                         client_res=client_res)

            ramaf = ramu.compute("AFTER EVALUATION")
            cpuaf = cpuu.compute("AFTER EVALUATION")
            if gpu_flag:
                gpuaf = gpuu.compute("AFTER EVALUATION")
            else:
                gpuaf = 0

            savecomp(output, strategy_name, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
            del strategy
            auto_garbage_collect()
        df2 = extract_metrics_gpu_csv(f"{output}/comp.csv")
        df2.to_csv(f"{output}/comp_extracted.csv")
        print(f"Run Completed for {n_cl} clients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for FL.")
    parser.add_argument("-s", "--strategy", type=str, default='all', help="Strategy to use")
    parser.add_argument("-m", "--model", type=str, default='MobileNet', help="Model to use")
    parser.add_argument("-n", "--rounds", type=int, default=10, help="Number of Rounds")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of Epochs")
    parser.add_argument("-x", "--batch_size", default=16, type=int, help="Batch Size?")
    parser.add_argument("-c", "--icl", type=int, default=2, help="Initial number of clients")
    parser.add_argument("-f", "--fcl", type=int, default=10, help="Final number of clients")
    parser.add_argument("-p", "--path", type=str, help="Path to dataset")
    parser.add_argument("-o", "--output", type=str, help="Output path")
    parser.add_argument("-a", "--aug", type=eval, choices=[True, False], default='False', help="Use Augmentation?")
    parser.add_argument("-b", "--base", type=str, default="FedAvg", help="Default base for FedRoot Only")
    parser.add_argument("-t", "--processor_type", type=str, default="cpu", help="Processor Type")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of action output heads (8=MANNERS-DB, 9=OfficeDB)")
    parser.add_argument("--group_col", type=str, default=None, help="Column assigning clients by group (robot/room) instead of random_split; #unique values must equal #clients")
    parser.add_argument("--split_col", type=str, default=None, help="Column with pre-computed train/test labels, used instead of the internal random 75/25 split")
    parser.add_argument("--action_cols", type=str, default=None, help="Comma-separated action column names (default: MANNERS-DB's 8 hardcoded names)")
    parser.add_argument("--extra_cols", type=str, default=None, help="Comma-separated pass-through id columns, e.g. group/task/split columns (default: 'Using circle,Using arrow')")
    parser.add_argument("--num_cpus", type=int, default=4, help="Total CPUs given to Ray's pool (set to the VM's vCPU count to enable concurrent clients; see OFFICEDB_MODIFICATIONS.md)")
    args = parser.parse_args()

    print("Running with the following arguments:")
    print("Strategy: ", args.strategy)
    print("Model: ", args.model)
    print("Rounds: ", args.rounds)
    print("Clients start: ", args.icl)
    print("Clients end: ", args.fcl)
    print("Path: ", args.path)
    print("Output: ", args.output)
    print("Augmentation: ", args.aug)
    print("Base: ", args.base)
    print("Processor Type: ", args.processor_type)
    auto_garbage_collect()
    run(args)