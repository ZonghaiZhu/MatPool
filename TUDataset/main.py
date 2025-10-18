# coding:utf-8
import os, time, argparse, random, torch
import numpy as np
import utils
from model import MatPool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset, Planetoid, Amazon
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', default='../../../GraphData/', type=str,
                    help="data path (dictionary)")
# TUDataset refer to https://chrsmrrs.github.io/datasets/docs/datasets/
parser.add_argument('--dataset', type=str, default="DD",
                    help='DD/PROTEINS/AIDS/NCI1/NCI109/FRANKENSTEIN/Mutagenicity/COIL-DEL/COIL-RAG')
parser.add_argument('--gnn', type=str, default='pem', help='gin, or gcn, pgcn or pem(default: gin)')
parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')

## For neural network
parser.add_argument('--permutation', type=bool, default=True,
                    help='True: permutation invariance model, False: permutation sensitive model')
parser.add_argument('--use_edge_attr', type=bool, default=False,
                    help='True: aggregate edge attribute for node, False: no use')
parser.add_argument('--num_layer', type=int, default=3,
                    help='number of GNN message passing layers (default: 3)')
parser.add_argument('--emb_dim', type=int, default=128, help='hidden size for node feature')

# Fro training
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_decay_epoch', default=10, type=int)
parser.add_argument('--lr_decay_rate', default=0.95, type=float)
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
# parser.add_argument('--dropout', type=float, default=0.3, help='weight decay') # saved property
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--least_epoch', type=int, default=30, help='maximum number of epochs')
parser.add_argument('--early_stop', type=int, default=20, help='patience for early stopping')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument("--run_times", type=int, default=10, help="seed for initializing training.")
parser.add_argument("--folds", type=int, default=10, help="10-folds cross-validation for training.")
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def K_fold(folds, data, seed):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(len(data)), data.data.y):
        test_indices.append(torch.tensor(idx, dtype=torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def main(dataset):
    args = parser.parse_args()
    args.dataset = dataset
    # prepare related data
    dataset = TUDataset(args.data_path, args.dataset, use_node_attr=True)
    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features
    args.num_edge_attr = dataset.num_edge_labels

    # prepare related documents
    print('\nSetting environment...')

    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = 'log/' + args.dataset + '_' + args.gnn + '_PI_' + str(args.permutation) + '_WE_' \
              + str(args.use_edge_attr) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir)
    utils.configure_output_dir(log_dir)

    # hyperparameters: lr, dropout, weight_decay, emb_dim, permutation
    lrs = [0.001, 0.0001]
    wds = [0]#not used
    emb_dims = [256]
    combinations = [{'lr': lr, 'wd': wd, 'emb_dim': emb_dim}
                    for lr in lrs for wd in wds for emb_dim in emb_dims]

    results = []
    for combination in combinations:
        args.lr = combination['lr']
        args.weight_decay = combination['wd']
        args.emb_dim = combination['emb_dim']

        args.num_features = dataset.num_features if dataset.num_features != 0 else args.emb_dim // 2

        train_scores, valid_scores, test_scores, epoch_times = [], [], [], []
        for run in range(args.run_times):
            # set random seed form 0 to 9
            random.seed(run)
            np.random.seed(run)
            torch.cuda.manual_seed(run)
            torch.random.manual_seed(run)

            # n-folds cross-validation, one fold for test, one fold for validation, and the other folds for training
            train_score, valid_score, test_score, epoch_time = 0, 0, 0, 0
            for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*K_fold(args.folds, dataset, run))):
                train_dataset = dataset[train_idx]
                test_dataset = dataset[test_idx]
                val_dataset = dataset[val_idx]

                train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
                valid_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

                # prepare model
                model = MatPool(args)

                # start training
                temp_train_score, temp_valid_score, temp_epoch_time = model.fit(train_loader, valid_loader, args.epochs)

                # start testing
                temp_test_score = model.predict(test_loader)
                del model

                train_score += temp_train_score
                valid_score += temp_valid_score
                test_score += temp_test_score

            train_scores.append(train_score/args.folds)
            valid_scores.append(valid_score/args.folds)
            test_scores.append(test_score/args.folds)
            epoch_times.append(epoch_time/args.folds)

        train_score_mean = round(np.mean(train_scores), 4)
        train_score_std = round(np.std(train_scores), 4)

        valid_score_mean = round(np.mean(valid_scores), 4)
        valid_score_std = round(np.std(valid_scores), 4)

        test_score_mean = round(np.mean(test_scores), 4)
        test_score_std = round(np.std(test_scores), 4)

        epoch_time_mean = round(np.mean(epoch_times), 4)

        temp = np.array([args.lr, args.weight_decay, args.emb_dim, epoch_time_mean,
                         train_score_mean, train_score_std, valid_score_mean, valid_score_std, test_score_mean,
                         test_score_std])
        results.append(temp)

    temp_results = np.array(results, dtype=np.float32)
    best_valid_idx = np.argmax(temp_results[:, -4])
    best_result = temp_results[best_valid_idx, :]
    print(('Mean test Score:{:.4f}, Std test score:{:.4f}').format(best_result[-2], best_result[-1]))

    # record classification results
    np.savetxt(log_dir + '/results.csv', temp_results, fmt='%.05f')
    # record classification results
    records = ['lr', 'weight_decay', 'emb_dim', 'epoch_time', 'train_score', 'train_std',
               'valid_score', 'valid_std', 'test_score', 'test_std']
    result_file = open(os.path.join(log_dir, "best_result.txt"), 'w')
    for val in zip(records, best_result):
        result_file.write(val[0] + ':' + np.array2string(val[1]) + '\n')
    result_file.close()


if __name__ == '__main__':
    datasets = ['AIDS', 'DD', 'PROTEINS', 'Letter-high', 'Letter-low', 'Letter-med',
                'NCI1', 'NCI109', 'FRANKENSTEIN', 'Mutagenicity', 'COIL-DEL', 'COIL-RAG']
    datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'COLORS-3'] # , 'MCF-7', 'MOLT-4'
    for dataset in datasets:
        main(dataset)