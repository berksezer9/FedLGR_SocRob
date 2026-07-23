from dataloader.utils import load_datasets_pretrain
from models.deepLabMobileNet import Net as deepNet
from models.MobileNet import Net as mobNet
import torch
import argparse
import pickle
#add utils from previous folder
import sys
sys.path.append('../')
from utils import train, test
def Average(lst):
    return sum(lst) / len(lst)
def run(args):
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

    # action_cols/extra_cols=None (default) reproduce MANNERS-DB's 8 hardcoded
    # action names + 'Using circle'/'Using arrow'; OfficeDB adaptation passes
    # --action_cols/--extra_cols (see OFFICEDB_MODIFICATIONS.md). Without
    # --extra_cols, load_images() looks for 'Using circle'/'Using arrow' in
    # every row regardless of the actual CSV schema -- on OfficeDB's
    # Robot/Room/Split-shaped all_data.csv that KeyErrors per-row inside
    # load_images' try/except, silently producing an empty dataframe.
    action_cols = args.action_cols.split(',') if args.action_cols else None
    extra_cols = args.extra_cols.split(',') if args.extra_cols else None
    data=load_datasets_pretrain(num_clients=1, split= args.split_ratio, batch_size=args.batch_size, path=args.data, aug=False, DEVICE=DEVICE,
                                 action_cols=action_cols, extra_cols=extra_cols)
    for i in range(len(models)):
        model=models[i]
        model_n=names[i]
        train(model=model, train_loader=data, epochs=args.epochs, DEVICE=DEVICE)
        y_labels=list(range(args.num_classes))
        a,b,c=test(net=model, testloader=data,y_labels=y_labels, DEVICE=DEVICE)
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
    args = parser.parse_args()

    print("Running with following arguments:")
    print("Training Model: {}".format(args.models))
    print("Split Ratio: {}".format(args.split_ratio))
    print("Batch Size: {}".format(args.batch_size))
    print("Epochs: {}".format(args.epochs))
    print("Path: {}".format(args.path))

    run(args)