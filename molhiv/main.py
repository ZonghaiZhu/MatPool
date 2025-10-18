# coding:utf-8
import os, time, argparse, random, torch
import numpy as np
import utils
from model import MatPool
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--data_path', default='../../../GraphData/', type=str,
                    help="data path (dictionary)")
parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                    help='dataset name (default: ogbg-molhiv)')
parser.add_argument('--gnn', type=str, default='pem', help='gin, gcn, pgcn, or pem(default: gin)')
parser.add_argument('--feature', type=str, default="full",
                    help='full feature or simple feature')

## For neural network
parser.add_argument('--permutation', type=bool, default=True,
                    help='True: permutation invariance model, False: permutation sensitive model')
parser.add_argument('--use_edge_attr', type=bool, default=False,
                    help='True: aggregate edge attribute for node, False: no use')
parser.add_argument('--num_layer', type=int, default=3,
                    help='number of GNN message passing layers (default: 2)')
parser.add_argument('--emb_dim', type=int, default=128, help='hidden size for node feature')

# Fro training
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--lr_decay_epoch', default=10, type=int)
parser.add_argument('--lr_decay_rate', default=0.95, type=float)
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
# parser.add_argument('--dropout', type=float, default=0.3, help='weight decay') # saved property
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--least_epoch', type=int, default=20, help='maximum number of epochs')
parser.add_argument('--early_stop', type=int, default=15, help='patience for early stopping')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument("--run_times", type=int, default=10, help="seed for initializing training.")
parser.add_argument('--device', default='cuda', type=str, help='use GPU.')


def main():
    args = parser.parse_args()
    # prepare related data
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.data_path)
    args.num_tasks = dataset.num_tasks

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # prepare related documents
    print('\nSetting environment...')

    if not os.path.exists('log'):
        os.makedirs('log')
    log_dir = 'log/' + args.dataset + '_' + args.gnn + '_PI_' + str(args.permutation) + '_WE_' \
              + str(args.use_edge_attr) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir)
    utils.configure_output_dir(log_dir)

    # hyperparameters: lr, dropout, weight_decay, emb_dim, permutation
    lrs = [0.0001]
    wds = [0] # not used
    emb_dims = [256]
    combinations = [{'lr':lr, 'wd':wd, 'emb_dim':emb_dim}
                    for lr in lrs for wd in wds for emb_dim in emb_dims]

    results = []
    for combination in combinations:
        args.lr = combination['lr']
        args.weight_decay = combination['wd']
        args.emb_dim = combination['emb_dim']

        train_scores, valid_scores, test_scores, epoch_times = [], [], [], []
        for run in range(args.run_times):
            # set random seed form 0 to 9
            random.seed(run)
            np.random.seed(run)
            torch.cuda.manual_seed(run)
            torch.random.manual_seed(run)

            # prepare model
            model = MatPool(args)

            # start training
            train_score, valid_score, epoch_time = model.fit(train_loader, valid_loader, args.epochs, dataset.task_type)

            # start testing
            test_score = model.predict(test_loader)['rocauc']

            train_scores.append(train_score)
            valid_scores.append(valid_score)
            test_scores.append(test_score)
            epoch_times.append(epoch_time)

            del model

        trn_val_tst = np.array([train_scores, valid_scores, test_scores]).T
        # record classification results
        np.savetxt(log_dir + '/' + str(args.lr) + '_' + str(args.emb_dim) + '.csv', trn_val_tst, fmt='%.05f')

        train_score_mean = round(np.mean(train_scores), 4)
        train_score_std = round(np.std(train_scores), 4)

        valid_score_mean = round(np.mean(valid_scores), 4)
        valid_score_std = round(np.std(valid_scores), 4)

        test_score_mean = round(np.mean(test_scores), 4)
        test_score_std = round(np.std(test_scores), 4)

        epoch_time_mean = round(np.mean(epoch_times), 4)

        temp = np.array([args.lr, args.weight_decay, args.emb_dim, epoch_time_mean,
                         train_score_mean, train_score_std, valid_score_mean, valid_score_std, test_score_mean, test_score_std])
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
        result_file.write(val[0] + ':' + np.array2string(val[1])+'\n')
    result_file.close()


if __name__ == '__main__':
    for i in range(3):
        main()
