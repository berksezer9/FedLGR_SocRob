"""OfficeDB -> Home transfer-learning eval: loads an already-trained
checkpoint (e.g. pretrain.py's output), evaluates it zero-shot on a
different dataset, and optionally fine-tunes on that dataset's train split
before re-evaluating. Added for the officedb-adapt fork's "office -> home
generalization" experiment -- see
fedlgr_officedb/OFFICEDB_MODIFICATIONS.md item 13 and
fedlgr_officedb/transfer_office_to_home.py (the thin driver that calls this,
mirroring pretrain_officedb.py -> pretrain.py).

Mirrors pretrain.py's structure (same load_datasets_pretrain-adjacent
imports, same --num_classes/--action_cols/--extra_cols generalization) but
uses dataloader.utils.load_datasets (num_clients=1) instead of
load_datasets_pretrain, since this needs a real held-out test split
(--split_col) rather than pretrain.py's single train==test split_ratio
slice.
"""
from dataloader.utils import load_datasets
from models.MobileNet import Net as mobNet
from models.deepLabMobileNet import Net as deepNet
import torch
import argparse
import json
import pickle
import sys
sys.path.append('../')
from utils import train, test


def run(args):
    DEVICE = torch.device('cuda' if args.processor == 'gpu' else 'cpu')

    if args.model == 'DeepLabMobileNet':
        model = deepNet(num_classes=args.num_classes)
    else:
        model = mobNet(num_classes=args.num_classes)

    with open(args.checkpoint, 'rb') as f:
        model.load_state_dict(pickle.load(f))

    action_cols = args.action_cols.split(',') if args.action_cols else None
    extra_cols = args.extra_cols.split(',') if args.extra_cols else None

    trainloaders, _testloaders_per_client, testloader, y_labels, _ = load_datasets(
        num_clients=1, path=args.data, aug=args.aug, batch_size=args.batch_size,
        DEVICE=DEVICE, action_cols=action_cols, extra_cols=extra_cols, split_col='Split')

    results = {}
    loss, pcc, rmse = test(net=model, testloader=testloader, y_labels=y_labels, DEVICE=DEVICE)
    print("Zero-shot:", loss, pcc, rmse)
    results['zero_shot'] = {'loss': loss, 'pcc': pcc, 'rmse': rmse}

    if args.finetune_epochs > 0:
        train(model=model, train_loader=trainloaders[0], epochs=args.finetune_epochs, DEVICE=DEVICE)
        loss, pcc, rmse = test(net=model, testloader=testloader, y_labels=y_labels, DEVICE=DEVICE)
        print(f"Fine-tuned ({args.finetune_epochs} epochs):", loss, pcc, rmse)
        results['finetuned'] = {'epochs': args.finetune_epochs, 'loss': loss, 'pcc': pcc, 'rmse': rmse}

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to a pretrain.py-produced .pkl checkpoint')
    parser.add_argument('--data', required=True, help='Path to target-domain prepared dataset (Split col required)')
    parser.add_argument('--output', required=True, help='Path to write results JSON')
    parser.add_argument('--model', default='MobileNet', choices=['MobileNet', 'DeepLabMobileNet'])
    parser.add_argument('--finetune_epochs', type=int, default=0,
                         help='If >0, fine-tune on the target train split before a second eval pass')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--aug', type=eval, choices=[True, False], default='False')
    parser.add_argument('--processor', default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--action_cols', type=str, default=None)
    parser.add_argument('--extra_cols', type=str, default=None)
    args = parser.parse_args()

    run(args)
