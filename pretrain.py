from dataloader.utils import load_datasets_pretrain, load_datasets, load_val_loader
from models.deepLabMobileNet import Net as deepNet
from models.MobileNet import Net as mobNet
from models.ResNet50 import Net as resNet50
import torch
import argparse
import pickle
import random
import numpy as np
#add utils from previous folder
import sys
sys.path.append('../')
from utils import train, test, train_with_early_stopping
def Average(lst):
    return sum(lst) / len(lst)
def run(args):
    # Multi-seed reruns (domain-transfer single-seed-reliability check,
    # docs/domain_transfer_officedb_to_mannersdbplus.md "Literature
    # verification" section) need model init + DataLoader shuffling to
    # actually vary per --seed and be reproducible for a given seed -- seed
    # every RNG source before anything stochastic happens (data split
    # ordering, model weight init, train/val shuffling). --seed is optional
    # and unset by default so all prior non-seeded runs/checkpoints are
    # unaffected.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    #load the datasets
    if args.processor=='cpu':
        DEVICE = torch.device('cpu')
        #add "cpu" to the path
        args.path = args.path + "/cpu"
        gpu = False
    else:
        DEVICE = torch.device('cuda')
        #add "gpu" to the path
        args.path = args.path + "/gpu"
        gpu = True

    if args.models == 'all':
        models=[deepNet(num_classes=args.num_classes), mobNet(num_classes=args.num_classes)]
        names=['DeepLabMobileNet', 'MobileNet']
    elif args.models == 'DeepLabMobileNet':
        models=[deepNet(num_classes=args.num_classes)]
        names=['DeepLabMobileNet']
    elif args.models == 'MobileNet':
        models=[mobNet(num_classes=args.num_classes)]
        names=['MobileNet']
    elif args.models == 'ResNet50':
        models=[resNet50(num_classes=args.num_classes)]
        names=['ResNet50']

    # action_cols/extra_cols=None (default) reproduce MANNERS-DB's 8 hardcoded
    # action names + 'Using circle'/'Using arrow'; OfficeDB adaptation passes
    # --action_cols/--extra_cols (see OFFICEDB_MODIFICATIONS.md). Without
    # --extra_cols, load_images() looks for 'Using circle'/'Using arrow' in
    # every row regardless of the actual CSV schema -- on OfficeDB's
    # Robot/Room/Split-shaped all_data.csv that KeyErrors per-row inside
    # load_images' try/except, silently producing an empty dataframe.
    action_cols = args.action_cols.split(',') if args.action_cols else None
    extra_cols = args.extra_cols.split(',') if args.extra_cols else None
    if args.split_col:
        # Leakage-safe path: honor a real train/test Split column (same
        # load_datasets() transfer_eval.py/main.py already use) instead of
        # load_datasets_pretrain's random split_ratio shuffle -- that
        # function ignores any pre-defined split and (see below) trains and
        # evaluates on the identical slice, which is wrong for any dataset
        # that ships its own leakage-safe splits (OFFICEDB_MODIFICATIONS.md
        # item 18).
        trainloaders, _testloaders, testloader, _y, _perm = load_datasets(
            num_clients=1, path=args.data, aug=False, batch_size=args.batch_size, DEVICE=DEVICE,
            action_cols=action_cols, extra_cols=extra_cols, split_col=args.split_col)
        train_loader = trainloaders[0]
        eval_loader = testloader
        val_loader = load_val_loader(
            args.data, batch_size=args.batch_size, action_cols=action_cols, extra_cols=extra_cols,
            split_col=args.split_col) if args.early_stopping else None
    else:
        # Original behavior, unchanged: no split_col means no leakage-safe
        # split is available, so fall back to load_datasets_pretrain's
        # random split_ratio slice, trained and evaluated on the same data
        # (this was already true before item 18 -- see its
        # OFFICEDB_MODIFICATIONS.md entry for why that's only acceptable
        # here, not for a real held-out eval).
        train_loader = load_datasets_pretrain(num_clients=1, split=args.split_ratio, batch_size=args.batch_size,
                                               path=args.data, aug=False, DEVICE=DEVICE,
                                               action_cols=action_cols, extra_cols=extra_cols)
        eval_loader = train_loader
        val_loader = None
    if args.early_stopping and val_loader is None:
        sys.exit("--early_stopping requires --split_col with 'val' rows in the data (none found)")
    for i in range(len(models)):
        model=models[i]
        model_n=names[i]
        y_labels=list(range(args.num_classes))
        if args.early_stopping:
            best_epoch, best_val_loss = train_with_early_stopping(
                model=model, train_loader=train_loader, val_loader=val_loader, DEVICE=DEVICE,
                y_labels=y_labels, max_epochs=args.max_epochs, patience=args.patience,
                lr=args.lr, clip_grad_norm=args.clip_grad)
            print(f"Early-stopped at epoch {best_epoch} (val loss {best_val_loss})")
        else:
            train(model=model, train_loader=train_loader, epochs=args.epochs, DEVICE=DEVICE)
        a,b,c,_,_,_=test(net=model, testloader=eval_loader,y_labels=y_labels, DEVICE=DEVICE)
        print(a, b, c)
        with open(args.path + "/" + model_n + ".pkl", 'wb') as f:
            pickle.dump(model.state_dict(), f)
        with open(args.path + "/" + model_n + ".pkl", 'rb') as f:
            model.load_state_dict(pickle.load(f))




if __name__ == "__main__":
    #define parser arguments, we need a model, split_ratio and batch_size
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, default='all', help='Model to use for training')
    parser.add_argument('-s', '--split_ratio', type=float, default=0.33, help='Split ratio for training')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
    #path to path of data
    parser.add_argument('-d', '--data', type=str, default='SADRA-Dataset', help='Path to data')
    parser.add_argument('-p', '--path', type=str, default='models', help='Path to save the model')
    #processor
    parser.add_argument('-t', '--processor', type=str, default='cpu', help='Processor to run the scrip')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of action output heads (8=MANNERS-DB, 9=OfficeDB)')
    parser.add_argument('--action_cols', type=str, default=None, help='Comma-separated action column names (default: MANNERS-DB\'s 8 hardcoded names)')
    parser.add_argument('--extra_cols', type=str, default=None, help='Comma-separated pass-through id columns, e.g. group/task/split columns (default: \'Using circle,Using arrow\')')
    parser.add_argument('--split_col', type=str, default=None,
                         help='If given and present in the data, train only on rows where this column == "train" '
                              '(load_datasets(), leakage-safe) instead of load_datasets_pretrain\'s random '
                              'split_ratio shuffle. Omit to preserve the original behavior exactly.')
    parser.add_argument('--early_stopping', action='store_true',
                         help='Train up to --max_epochs, evaluating a held-out "val" split (--split_col) '
                              'after every epoch, and keep the lowest-val-loss epoch\'s weights instead of '
                              'whatever --epochs happened to land on. Requires --split_col and \'val\' rows '
                              'in the data. Default off: reproduces the original fixed --epochs behavior.')
    parser.add_argument('--max_epochs', type=int, default=40,
                         help='Epoch cap when --early_stopping (default 40; ignored otherwise)')
    parser.add_argument('--patience', type=int, default=5,
                         help='Early-stopping patience in epochs (default 5; ignored unless --early_stopping)')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed random/numpy/torch before model init and training, for multi-seed '
                              'reruns. Default None: unseeded (original behavior, unchanged).')
    parser.add_argument('--lr', type=float, default=0.001,
                         help='Adam learning rate for --early_stopping training (default 0.001, the '
                              'original hardcoded value). CNN domain-transfer arm passes 1e-4 -- see '
                              'OFFICEDB_MODIFICATIONS.md item 38.')
    parser.add_argument('--clip_grad', type=float, default=None,
                         help='If set, clip_grad_norm_ max-norm applied before optimizer.step() in '
                              '--early_stopping training (default None: no clipping, original behavior). '
                              'CNN domain-transfer arm passes 1.0 -- OFFICEDB_MODIFICATIONS.md item 38.')
    args = parser.parse_args()

    print("Running with following arguments:")
    print("Training Model: {}".format(args.models))
    print("Split Ratio: {}".format(args.split_ratio))
    print("Batch Size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.epochs))
    print("Path: {}".format(args.path))

    run(args)