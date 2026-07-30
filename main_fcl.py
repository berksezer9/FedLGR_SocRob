import argparse

from CL.default import EWC
from client.default import FlowerClientCL, FlowerClient_NR
from client.fedRoot import FlowerClientCL_Root, FlowerClient_NR_Root, FlowerClient_LGR
from models.GenNet import VAE
from dataloader.utils import task_splitter, load_datasets
from server.strategies import FedAvgWithAccuracyMetric
import ray
import flwr as fl
from server.utils import fit_config, evaluate_config
import torch
import matplotlib.pyplot as plt
from utils import get_parameters
import pandas as pd
import os
from CL.default import EWCOnline, SI, MAS, LatentGenerativeReplay
from CL.default import Naive_Rehearsal as NR
from datetime import datetime
from metrics.computation import RAMU, CPUUsage, GPUUsage
from utils import plot_results, get_eval_fn_cl, extract_metrics_gpu_csv, truncate_float
import gc
import pickle
import warnings

warnings.filterwarnings("ignore")


def run_strategy(strategy, strategy_name, coeff, client_fn, clients, rounds, epochs, output, aug, ray_init_args,
                 client_res, n_tasks=2):
    # Generalizes vendor's original hardcoded 2-task Loss1/RMSE1/PCC1/Loss2/RMSE2/PCC2
    # columns to N task-boundary columns (see OFFICEDB_MODIFICATIONS.md). For
    # n_tasks=2 this produces the identical column set/values as before.
    print("Running strategy " + str(strategy_name) + " for " + str(clients) + "clients!")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(clients),
        config=fl.server.ServerConfig(num_rounds=int(rounds)),  # Just three rounds
        strategy=strategy,
        ray_init_args=ray_init_args,
        client_resources=client_res,
    )
    rounds_per_task = int(rounds) // n_tasks
    # server_round number of each task's final round (1-indexed, matches Flower's config["server_round"])
    task_rounds = [rounds_per_task * k for k in range(1, n_tasks + 1)]
    metric_cols = []
    for t in range(1, n_tasks + 1):
        metric_cols += [f"Loss{t}", f"RMSE{t}", f"PCC{t}"]

    try:
        data = pd.read_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_decentral.csv")
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        data = pd.DataFrame(columns=["Method", "reg_coeff"] + metric_cols)

    row = [f"{strategy_name}", f"{coeff}"]
    for r in task_rounds:
        # losses_distributed/metrics_distributed are 0-indexed per completed round (round 1 -> index 0)
        idx = r - 1
        row += [truncate_float(history.losses_distributed[idx][-1], 4),
                truncate_float(history.metrics_distributed['avg_rmse'][idx][-1], 4),
                truncate_float(history.metrics_distributed['avg_pearson_score'][idx][-1], 4)]
    data = pd.concat([data, pd.Series(row, index=data.columns).to_frame().T])

    data.to_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_decentral.csv")

    try:
        try:
            data = pd.read_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_central.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data = pd.DataFrame(columns=["Method", "reg_coeff"] + metric_cols)

        row = [f"{strategy_name}", f"{coeff}"]
        for r in task_rounds:
            # losses_centralized/metrics_centralized include an initial round-0
            # evaluation (before training) at index 0, so round r sits at index r.
            row += [truncate_float(history.losses_centralized[r][-1], 4),
                    truncate_float(history.metrics_centralized['avg_rmse'][r][-1], 4),
                    truncate_float(history.metrics_centralized['avg_pearson_score'][r][-1], 4)]
        data = pd.concat([data, pd.Series(row, index=data.columns).to_frame().T])
        data.to_csv(f"{output}/{clients}_{rounds}_{epochs}_{aug}_central.csv")

    except:
        print("Centralised results not available")
    if '-' in strategy_name:
        strategy_name, base_name = strategy_name.split('-')
        if 'LGR' in base_name:
            save_path = f"{output}/{strategy_name}/{strategy_name}-{base_name}_{aug}_{clients}_{rounds}_{epochs}"
        else:
            save_path = f"{output}/{coeff}/{strategy_name}-{base_name}_{aug}_{clients}_{rounds}_{epochs}"
    else:
        save_path = f"{output}/{coeff}/{strategy_name}_{aug}_{clients}_{rounds}_{epochs}"

    plot_results(history=history, save_path=save_path)
    del history
    ray.shutdown()

def savecomp(output, strat, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf):
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
    def client_fn_reg(cid) -> FlowerClientCL:
        return FlowerClientCL(cid, net.to(DEVICE), trainloader=trainloaders, valloader=valloaders,
                              testloader=testloader, epochs=int(args.epochs),
                              y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds),
                              path=f"{output_reg}",
                              DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params, n_tasks=args.n_tasks,
                              active_idx_per_task=active_idx_per_task, cumulative_idx_per_task=cumulative_idx_per_task)

    def client_fn_NR(cid) -> FlowerClient_NR:
        return FlowerClient_NR(cid, net.to(DEVICE), trainloader=trainloaders, valloader=valloaders,
                               testloader=testloader, epochs=int(args.epochs),
                               y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds),
                               path=f"{output_NR}",
                               DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params, n_tasks=args.n_tasks,
                               active_idx_per_task=active_idx_per_task, cumulative_idx_per_task=cumulative_idx_per_task)

    def client_fn_reg_root(cid) -> FlowerClientCL_Root:
        return FlowerClientCL_Root(cid, net.to(DEVICE), trainloader=trainloaders, valloader=valloaders,
                                   testloader=testloader, epochs=int(args.epochs),
                                   y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds),
                                   path=f"{output_root_reg}",
                                   DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params, n_tasks=args.n_tasks,
                                   active_idx_per_task=active_idx_per_task, cumulative_idx_per_task=cumulative_idx_per_task)

    def client_fn_NR_root(cid) -> FlowerClient_NR_Root:
        return FlowerClient_NR_Root(cid, net.to(DEVICE), trainloader=trainloaders, valloader=valloaders,
                                    testloader=testloader, epochs=int(args.epochs),
                                    y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds),
                                    path=f"{output_root_NR}",
                                    DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params, n_tasks=args.n_tasks,
                                    active_idx_per_task=active_idx_per_task, cumulative_idx_per_task=cumulative_idx_per_task)

    def client_fn_LGR(cid) -> FlowerClient_LGR:
        return FlowerClient_LGR(cid, net=net.to(DEVICE), trainloader=trainloaders, valloader=valloaders,
                                testloader=testloader, gr=gr.to(DEVICE), epochs=int(args.epochs),
                                y_labels=y_labels, cl_strategy=caller, agent_config=agent_config, nrounds=int(args.rounds),
                                path=f"{output}",
                                DEVICE=DEVICE, num_clients=n_cl, strat_name=strat_cl, params=params, n_tasks=args.n_tasks,
                                active_idx_per_task=active_idx_per_task, cumulative_idx_per_task=cumulative_idx_per_task)

    if args.model == 'MobileNet':
        from models.MobileNet import Net
    elif args.model == 'DeepLabMobileNet':
        from models.deepLabMobileNet import Net

    # None (default) reproduces vendor's original MANNERS-DB column names/2-task
    # circle-arrow split; comma-separated CLI values override them for OfficeDB's
    # Axis A/B task splits (see OFFICEDB_MODIFICATIONS.md).
    action_cols = args.action_cols.split(',') if args.action_cols else None
    extra_cols = args.extra_cols.split(',') if args.extra_cols else None

    def _coerce(v):
        try:
            return float(v)
        except ValueError:
            return v
    task_values = [_coerce(v) for v in args.task_values.split(',')] if args.task_values else [1, 0]

    # FCL Axis A (action-subset): 'action_subset' mode doesn't row-filter by
    # a task column at all (every scene is relevant to every task -- see
    # OFFICEDB_MODIFICATIONS.md) -- it reuses the SAME single train/test
    # split for every task and instead masks the loss/eval to each task's
    # active output columns. active_idx_per_task[t] is task t's OWN
    # exclusive columns (used for training); cumulative_idx_per_task[t] is
    # the union of columns revealed by tasks 0..t (used for evaluation, per
    # the "train exclusive, evaluate cumulative" protocol). Both are None in
    # the default 'row_filter' mode, reproducing the original/Axis B
    # behaviour exactly.
    if args.axis_mode == 'action_subset':
        action_groups = [grp.split('|') for grp in args.action_groups.split(';')]
        active_idx_per_task = [[action_cols.index(a) for a in grp] for grp in action_groups]
        cumulative_idx_per_task = []
        seen = []
        for idxs in active_idx_per_task:
            seen = seen + [i for i in idxs if i not in seen]
            cumulative_idx_per_task.append(list(seen))
    else:
        active_idx_per_task = None
        cumulative_idx_per_task = None

    # make a directory in the output folder with name "YYMMDD_HHMMSS" followed by the arguments
    experiment_path = f"{args.output}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.strategy_fl}_{args.strategy_cl}_{args.model}_{args.rounds}_{args.icl}_{args.fcl}_{args.aug}_{args.processor_type}"
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

    # Running the loop for clients between icl (low) and fcl (high)
    for n_cl in range(int(args.icl), int(args.fcl) + 1):
        net = Net(num_classes=args.num_classes)
        params = get_parameters(net)
        # See main.py / OFFICEDB_MODIFICATIONS.md item 14: divide num_cpus by n_cl so
        # clients aren't each requesting Ray's entire CPU pool (which serializes them).
        if gpu_flag == 1:
            client_res = {"num_gpus": num_GPUs / n_cl, "num_cpus": num_CPUs / n_cl}
        else:
            client_res = {"num_gpus": num_GPUs, "num_cpus": num_CPUs / n_cl}

        if not os.path.exists(f"{args.output}/{n_cl}"):
            os.mkdir(f"{args.output}/{n_cl}")

        if args.axis_mode == 'action_subset':
            # Axis A: one shared partition reused across every task (not a
            # per-task row filter) -- see OFFICEDB_MODIFICATIONS.md and the
            # active_idx_per_task/cumulative_idx_per_task computation above.
            trainloaders_single, testloader_single, y_labels, _ = load_datasets(
                num_clients=n_cl, path=args.path, aug=args.aug, batch_size=args.batch_size,
                action_cols=action_cols, extra_cols=extra_cols, split_col=args.split_col)
            trainloaders = [trainloaders_single] * args.n_tasks
            valloaders = [testloader_single] * args.n_tasks
            testloader = testloader_single
        else:
            trainloaders, valloaders, testloader, y_labels = task_splitter(
                path=args.path, task_col=args.task_col, task_values=task_values, n_clients=n_cl, aug=args.aug,
                batch_size=args.batch_size, action_cols=action_cols, extra_cols=extra_cols, split_col=args.split_col)


        if args.strategy_fl == 'all':
            strategies = ['FedAvg', 'FedRoot']
        else:
            strategies = [args.strategy_fl]
        if args.strategy_cl == 'all':
            strategies_cl = ['EWC', 'EWCOnline', 'SI', 'MAS', 'NR', 'LGR']
        else:
            strategies_cl = [args.strategy_cl]

        for strat in strategies:
            path = f"{args.output}/{n_cl}/{strat}"
            if not os.path.exists(path):
                os.mkdir(path)
                
            if strat == 'FedAvg':
                for strat_cl in strategies_cl:
                    if strat_cl == 'LGR':
                        continue
                    print("Running FL Strategy: " + str(strat) + ": " + str(strat_cl))
                    
                    output = f"{args.output}/{n_cl}/{strat}/{strat_cl}"
                    if not os.path.exists(output):
                        os.mkdir(output)
                        
                    if strat_cl in ['EWC', 'EWCOnline', 'SI', 'MAS']:
                        if args.reg_coef == 'all':
                            reg_coefficients = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
                        else:
                            reg_coefficients = [float(args.reg_coef)]
                        
                        for coeff in reg_coefficients:
                            output_reg = f"{output}/{coeff}"
                            if not os.path.exists(f"{output_reg}"):
                                os.mkdir(f"{output_reg}")

                            agent_config = {'lr': 0.001, 'momentum': 0.1, 'weight_decay': 0.01,
                                            'schedule': [int(args.epochs)],
                                            'model_type': 'mode', 'model_name': 'model', 'model_weights': '',
                                            'out_dim': {'All': args.num_classes},
                                            'optimizer': 'Adam', 'print_freq': 0, 'gpuid': [gpu_flag],
                                            'reg_coef': coeff}

                            if strat_cl == 'EWC':
                                caller = EWC
                            elif strat_cl == 'EWCOnline':
                                caller = EWCOnline
                            elif strat_cl == 'SI':
                                caller = SI
                            elif strat_cl == 'MAS':
                                caller = MAS
                            client_fn = client_fn_reg

                            strategy = FedAvgWithAccuracyMetric(
                                min_available_clients=int(n_cl),
                                initial_parameters=fl.common.ndarrays_to_parameters(params),
                                on_fit_config_fn=fit_config,
                                on_evaluate_config_fn=evaluate_config,
                                evaluate_fn=get_eval_fn_cl(net, testloader=valloaders, DEVICE=DEVICE, y_labels=y_labels,
                                                            rounds_per_task=int(args.rounds) // args.n_tasks, n_tasks=args.n_tasks,
                                                            active_idx_per_task=cumulative_idx_per_task)
                            )
                            
                            rambef = ramu.compute("BEFORE EVALUATION")
                            cpubef = cpuu.compute("BEFORE EVALUATION")
                            if gpu_flag == 1:
                                gpubeff = gpuu.compute("BEFORE EVALUATION")
                            else:
                                gpubeff = 0
                            
                            run_strategy(strategy=strategy, strategy_name=f"{strat_cl}", coeff=coeff,
                                         client_fn=client_fn, clients=n_cl, rounds=int(args.rounds),
                                         epochs=int(args.epochs), output=f"{output}", aug=args.aug,
                                         ray_init_args=ray_init_args, client_res=client_res, n_tasks=args.n_tasks)
                            
                            ramaf = ramu.compute("AFTER EVALUATION")
                            cpuaf = cpuu.compute("AFTER EVALUATION")
                            if gpu_flag == 1:
                                gpuaf = gpuu.compute("AFTER EVALUATION")
                            else:
                                gpuaf = 0
                            savecomp(f"{output}", coeff, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
                    elif strat_cl == 'NR':
                        if args.reg_coef == 'all':
                            buffer_sizes = [1, 10, 100, 1000]
                        else:
                            buffer_sizes = [int(args.reg_coef)]
                            if buffer_sizes[0] == 0:
                                buffer_sizes = [1]
                        for buffer_size in buffer_sizes:
                            output_NR = f"{output}/{buffer_size}"
                            if not os.path.exists(f"{output_NR}"):
                                os.mkdir(f"{output_NR}")

                            agent_config = {'lr': 0.001, 'momentum': 0.1, 'weight_decay': 0.01,
                                            'schedule': [int(args.epochs)],
                                            'model_type': 'mode', 'model_name': 'model', 'model_weights': '',
                                            'out_dim': {'All': args.num_classes},
                                            'optimizer': 'Adam', 'print_freq': 0, 'gpuid': [gpu_flag],
                                            'memory_size': buffer_size, 'reg_coef': 0.01}
                            caller = NR
                            client_fn = client_fn_NR

                            strategy = FedAvgWithAccuracyMetric(
                                min_available_clients=int(n_cl),
                                initial_parameters=fl.common.ndarrays_to_parameters(params),
                                on_fit_config_fn=fit_config,
                                on_evaluate_config_fn=evaluate_config,
                                evaluate_fn=get_eval_fn_cl(net, testloader=valloaders, DEVICE=DEVICE, y_labels=y_labels,
                                                            rounds_per_task=int(args.rounds) // args.n_tasks, n_tasks=args.n_tasks,
                                                            active_idx_per_task=cumulative_idx_per_task)

                            )
                            rambef = ramu.compute("BEFORE EVALUATION")
                            cpubef = cpuu.compute("BEFORE EVALUATION")
                            if gpu_flag == 1:
                                gpubeff = gpuu.compute("BEFORE EVALUATION")
                            else:
                                gpubeff = 0
                            run_strategy(strategy, f"{strat_cl}", buffer_size, client_fn, n_cl, int(args.rounds),
                                         int(args.epochs),
                                         f"{output}", args.aug, ray_init_args, client_res, args.n_tasks)
                            ramaf = ramu.compute("AFTER EVALUATION")
                            cpuaf = cpuu.compute("AFTER EVALUATION")
                            if gpu_flag == 1:
                                gpuaf = gpuu.compute("AFTER EVALUATION")
                            else:
                                gpuaf = 0
                            savecomp(f"{output}", buffer_size, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
                    df2 = extract_metrics_gpu_csv(f"{output}/comp.csv")
                    df2.to_csv(f"{output}/comp_extracted.csv")
            elif strat == 'FedRoot':
                params = get_parameters(net.conv_module)
                
                for strat_cl in strategies_cl:
                    output = f"{args.output}/{n_cl}/{strat}/{strat_cl}"
                    if not os.path.exists(output):
                        os.mkdir(output)
                    if strat_cl in ['EWC', 'EWCOnline', 'SI', 'MAS']:
                        if args.reg_coef == 'all':
                            reg_coefficients = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000]
                        else:
                            reg_coefficients = [float(args.reg_coef)]
                        for coeff in reg_coefficients:
                            
                            output_root_reg = f"{output}/{coeff}"
                            if not os.path.exists(output_root_reg):
                                os.mkdir(f"{output_root_reg}")
                            agent_config = {'lr': 0.0001, 'momentum': 0.1, 'weight_decay': 0.01,
                                            'schedule': [int(args.epochs)],
                                            'model_type': 'mode', 'model_name': 'model', 'model_weights': '',
                                            'out_dim': {'All': args.num_classes}, 'optimizer':
                                                'Adam', 'print_freq': 0, 'gpuid': [gpu_flag], 'reg_coef': coeff}
                            if strat_cl == 'EWC':
                                caller = EWC
                            elif strat_cl == 'EWCOnline':
                                caller = EWCOnline
                            elif strat_cl == 'SI':
                                caller = SI
                            elif strat_cl == 'MAS':
                                caller = MAS
                            client_fn = client_fn_reg_root
                            strategy = FedAvgWithAccuracyMetric(
                                min_available_clients=int(n_cl),
                                initial_parameters=fl.common.ndarrays_to_parameters(params),
                                on_fit_config_fn=fit_config,
                                on_evaluate_config_fn=evaluate_config
                            )
                            rambef = ramu.compute("BEFORE EVALUATION")
                            cpubef = cpuu.compute("BEFORE EVALUATION")
                            if gpu_flag == 1:
                                gpubeff = gpuu.compute("BEFORE EVALUATION")
                            else:
                                gpubeff = 0
                            run_strategy(strategy=strategy, strategy_name=f"{strat_cl}", coeff=coeff,
                                         client_fn=client_fn, clients=n_cl, rounds=int(args.rounds),
                                         epochs=int(args.epochs), output=f"{output}", aug=args.aug,
                                         ray_init_args=ray_init_args, client_res=client_res, n_tasks=args.n_tasks)
                            ramaf = ramu.compute("AFTER EVALUATION")
                            cpuaf = cpuu.compute("AFTER EVALUATION")
                            if gpu_flag == 1:
                                gpuaf = gpuu.compute("AFTER EVALUATION")
                            else:
                                gpuaf = 0
                            savecomp(f"{output}", coeff, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
                    elif strat_cl == 'NR':
                        if args.reg_coef == 'all':
                            buffer_sizes = [1, 10, 100, 1000]
                        else:
                            buffer_sizes = [int(args.reg_coef)]
                        for buffer_size in buffer_sizes:
                            output_root_NR = f"{output}/{buffer_size}"
                            if not os.path.exists(output_root_NR):
                                os.mkdir(f"{output_root_NR}")
                            agent_config = {'lr': 0.0001, 'momentum': 0.1, 'weight_decay': 0.01,
                                            'schedule': [int(args.epochs)],
                                            'model_type': 'mode', 'model_name': 'model', 'model_weights': '',
                                            'out_dim': {'All': args.num_classes},
                                            'optimizer': 'Adam', 'print_freq': 0, 'gpuid': [gpu_flag],
                                            'memory_size': buffer_size, 'reg_coef': 0.01}
                            caller = NR

                            client_fn = client_fn_NR_root

                            strategy = FedAvgWithAccuracyMetric(
                                min_available_clients=int(n_cl),
                                initial_parameters=fl.common.ndarrays_to_parameters(params),
                                on_fit_config_fn=fit_config,
                                on_evaluate_config_fn=evaluate_config

                            )
                            rambef = ramu.compute("BEFORE EVALUATION")
                            cpubef = cpuu.compute("BEFORE EVALUATION")
                            if gpu_flag == 1:
                                gpubeff = gpuu.compute("BEFORE EVALUATION")
                            else:
                                gpubeff = 0
                            run_strategy(strategy, f"{strat_cl}", buffer_size, client_fn, n_cl, int(args.rounds),
                                         int(args.epochs),
                                         f"{output}", args.aug, ray_init_args, client_res, args.n_tasks)
                            ramaf = ramu.compute("AFTER EVALUATION")
                            cpuaf = cpuu.compute("AFTER EVALUATION")
                            if gpu_flag == 1:
                                gpuaf = gpuu.compute("AFTER EVALUATION")
                            else:
                                gpuaf = 0
                            savecomp(f"{output}", buffer_size, rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
                    elif strat_cl == 'LGR':
                        agent_config = {'lr': 0.001, 'momentum': 0.1, 'weight_decay': 0.01,
                                        'schedule': [int(args.epochs)],
                                        'model_type': 'mode', 'model_name': 'model', 'model_weights': '',
                                        'out_dim': {'All': args.num_classes},
                                        'optimizer': 'Adam', 'print_freq': 0, 'gpuid': [gpu_flag], 'reg_coef': 0.01}

                        caller = LatentGenerativeReplay

                        output_root_LGR = f"{output}/{strat_cl}"
                        if not os.path.exists(output_root_LGR):
                            os.mkdir(f"{output_root_LGR}")
                        # pretrained_dir defaults to 'models', reproducing the
                        # original cwd-relative path exactly; OfficeDB adaptation
                        # passes --pretrained_dir so this doesn't depend on cwd
                        # (see OFFICEDB_MODIFICATIONS.md).
                        if args.model == 'MobileNet':
                            input_dim = 1280  # Replace with the size of your input data
                            if gpu_flag == 1:
                                with open(f'{args.pretrained_dir}/gpu/MobileNet.pkl', 'rb') as f:
                                    net.load_state_dict(pickle.load(f), strict=True)
                                print("-------------------Loaded pretrained MobileNet for GPU-------------------")
                            else:
                                with open(f'{args.pretrained_dir}/cpu/MobileNet.pkl', 'rb') as f:
                                    net.load_state_dict(pickle.load(f), strict=True)
                                print("-------------------Loaded pretrained MobileNet for CPU-------------------")
                        elif args.model == 'DeepLabMobileNet':
                            input_dim = 1344
                            if gpu_flag == 1:
                                with open(f'{args.pretrained_dir}/gpu/DeepLabMobileNet.pkl', 'rb') as f:
                                    net.load_state_dict(pickle.load(f), strict=True)
                                print("-------------------Loaded pretrained DeepLabMobileNet for GPU-------------------")
                            else:
                                with open(f'{args.pretrained_dir}/cpu/DeepLabMobileNet.pkl', 'rb') as f:
                                    net.load_state_dict(pickle.load(f), strict=True)
                                print("-------------------Loaded pretrained DeepLabMobileNet for CPU-------------------")
                        # params = get_parameters(net.conv_module)
                        latent_dim = 64  # Set according to your desired latent space dimension
                        encoder_units = [256, 128]  # Adjust as needed
                        decoder_units = [128, 256]  # Adjust as needed

                        gr = VAE(input_dim, latent_dim, encoder_units, decoder_units)
                        client_fn = client_fn_LGR
                        strategy = FedAvgWithAccuracyMetric(
                            min_available_clients=int(n_cl),
                            initial_parameters=fl.common.ndarrays_to_parameters(params),
                            on_fit_config_fn=fit_config,
                            on_evaluate_config_fn=evaluate_config

                        )
                        rambef = ramu.compute("BEFORE EVALUATION")
                        cpubef = cpuu.compute("BEFORE EVALUATION")
                        if gpu_flag == 1:
                            gpubeff = gpuu.compute("BEFORE EVALUATION")
                        else:
                            gpubeff = 0
                        run_strategy(strategy, f"{strat_cl}", 'LGR', client_fn, n_cl, int(args.rounds),
                                     int(args.epochs),
                                     f"{output}", args.aug, ray_init_args, client_res, args.n_tasks)
                        ramaf = ramu.compute("AFTER EVALUATION")
                        cpuaf = cpuu.compute("AFTER EVALUATION")
                        if gpu_flag == 1:
                            gpuaf = gpuu.compute("AFTER EVALUATION")
                        else:
                            gpuaf = 0
                        savecomp(f"{output}", 'LGR', rambef, ramaf, cpubef, cpuaf, gpubeff, gpuaf)
                    df2 = extract_metrics_gpu_csv(f"{output}/comp.csv")
                    df2.to_csv(f"{output}/comp_extracted.csv")


# run for FedAvgEWC for 2 clients 1 round 1 epoch
# python main.py -sfl FedAvg -scl EWC -m MobileNet -n 2 -e 1 -c 2 -f 2 -p SARDA-Dataset -o output -a False -pro cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for FCL.")
    parser.add_argument("-sfl", "--strategy_fl", type=str, default='all', help="FL Strategy to use")
    parser.add_argument("-scl", "--strategy_cl", type=str, default='all', help="CL Strategy to use")
    parser.add_argument("-m", "--model", type=str, default='MobileNet', help="Model to use")
    parser.add_argument("-r", "--reg_coef", type=str, default='all', help="Regularisation Coefficient.")
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
    parser.add_argument("--n_tasks", type=int, default=2, help="Number of sequential FCL tasks (2=MANNERS-DB circle/arrow, 3=OfficeDB Axis A, 6=OfficeDB Axis B); rounds must be divisible by n_tasks")
    parser.add_argument("--task_col", type=str, default="Using circle", help="Column whose values define task membership")
    parser.add_argument("--task_values", type=str, default=None, help="Comma-separated ordered task values matching --task_col (default: '1,0', i.e. circle then arrow)")
    parser.add_argument("--pretrained_dir", type=str, default="models", help="Directory holding {cpu,gpu}/<model>.pkl pretrained checkpoints for LGR (was a bare cwd-relative 'models' path)")
    parser.add_argument("--action_cols", type=str, default=None, help="Comma-separated action column names (default: MANNERS-DB's 8 hardcoded names)")
    parser.add_argument("--extra_cols", type=str, default=None, help="Comma-separated pass-through id columns, e.g. group/task/split columns (default: 'Using circle,Using arrow')")
    parser.add_argument("--split_col", type=str, default=None, help="Column with pre-computed train/test labels, used instead of the internal random 75/25 split per task")
    parser.add_argument("--axis_mode", type=str, default="row_filter", choices=["row_filter", "action_subset"],
                         help="'row_filter' (default): task_col/task_values row-filtering (MANNERS-DB circle/arrow, OfficeDB Axis B). "
                              "'action_subset': OfficeDB Axis A -- every scene is relevant to every task; one shared train/test split is "
                              "reused across all tasks and --action_groups selects which output columns are active per task instead.")
    parser.add_argument("--action_groups", type=str, default=None,
                         help="Only used when --axis_mode action_subset. Per-task active action groups: tasks separated by ';', "
                              "action names within a task separated by '|' (names must be a subset of --action_cols). "
                              "E.g. 'A|B|C;D|E|F;G|H|I' for 3 tasks of 3 actions each.")
    parser.add_argument("--num_cpus", type=int, default=4, help="Total CPUs given to Ray's pool (set to the VM's vCPU count to enable concurrent clients; see OFFICEDB_MODIFICATIONS.md)")
    args = parser.parse_args()

    print("Running with the following arguments:")
    print("Strategy FL: ", args.strategy_fl)
    print("Strategy CL: ", args.strategy_cl)
    print("Model: ", args.model)
    print("Rounds: ", args.rounds)
    print("Epochs: ", args.epochs)
    print("Batch_size: ", args.batch_size)
    print("Clients start: ", args.icl)
    print("Clients end: ", args.fcl)
    print("Path: ", args.path)
    print("Output: ", args.output)
    print("Augmentation: ", args.aug)
    print("Processor Type: ", args.processor_type)

    run(args)

# Run FedAvg EWC with 2 clients 2 rounds 1 epoch
# python main_fcl.py -sfl FedRoot -scl LGR -m MobileNet -r 1000 -n 2 -e 1 -c 2 -f 2 -p SADRA-Dataset -o output -a False -t cpu -b FedAvg
