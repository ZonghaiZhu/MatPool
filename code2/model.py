# coding:utf-8
import torch, time, copy
import numpy as np
import torch.nn as nn
from net import Net
import utils
from tqdm import tqdm
from ogb.graphproppred import Evaluator
from utils import decode_arr_to_seq


class MatPool():
    def __init__(self, args, idx2vocab, node_encoder):
        self.args = args
        self.idx2vocab = idx2vocab
        self.print_freq = args.print_freq
        self.device = args.device
        self.net = Net(args, node_encoder).to(self.device)
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

        num_params = sum(p.numel() for p in self.net.parameters())
        print(f'#Params: {num_params}')

        self.multicls_criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator(args.dataset)

    def fit(self, train_loader, valid_loader, epochs):
        # switch to train mode
        self.net.train()
        best_train_score = 0
        best_valid_score = 0
        best_epoch = 0

        epoch_times = []
        for epoch in range(epochs):
            start_time = time.time()

            # adjust learning rate
            if epoch > self.args.lr_decay_epoch:
                new_lr = self.args.lr * pow(self.args.lr_decay_rate, (epoch - self.args.lr_decay_epoch))
                new_lr = max(new_lr, 1e-5)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr

            losses = []
            seq_ref_list = []
            seq_pred_list = []
            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                batch = batch.to(self.device)
                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    pass
                else:
                    pred_list = self.net(batch)
                    self.optimizer.zero_grad()
                    loss = 0
                    for i in range(len(pred_list)):
                        loss += self.multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])
                    loss = loss / len(pred_list)
                    loss.backward()
                    self.optimizer.step()

                # mat = []
                # for i in range(len(pred_list)):
                #     mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1)) # torch.argmax without backward already
                # mat = torch.cat(mat, dim=1)
                #
                # seq_pred = [self.arr_to_seq(arr) for arr in mat]
                # seq_ref = [batch.y[i] for i in range(len(batch.y))]
                #
                # seq_ref_list.extend(seq_ref)
                # seq_pred_list.extend(seq_pred)

                losses.append(loss.item())

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            # validation process
            # input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
            # train_score = self.evaluator.eval(input_dict)['F1']
            train_score = 0
            valid_score = self.predict(valid_loader)['F1']

            if valid_score > best_valid_score:
                best_net = copy.deepcopy(self.net)
                best_valid_score = valid_score
                best_train_score = train_score
                best_epoch = epoch
            else:
                if epoch >= self.args.least_epoch and epoch - best_epoch > self.args.early_stop:
                    print('\nEarly stop at %d epoch. The best is in %d epoch' %
                          (epoch, best_epoch))
                    self.net = best_net
                    break

            if epoch % self.args.print_freq == 0:
                print(('Epoch:[{}/{}], Epoch_time:{:.3f}, Train_loss:{:.4f},Train_Score:{:.4f}, Valid_Score:{:.4f}').format(
                    epoch + 1, self.args.epochs, epoch_time, np.mean(losses), train_score, valid_score))

            utils.log_tabular("Epoch", epoch)
            utils.log_tabular("Training_Time", np.mean(epoch_times))
            utils.log_tabular("Train_Loss", np.mean(losses))
            utils.log_tabular("Train_Score", train_score)
            utils.log_tabular("Valid_Score", valid_score)
            utils.dump_tabular()

        # write -1 to represent the end of one combination of hyperparameters
        utils.log_tabular("Epoch", -1)
        utils.log_tabular("Training_Time", -1)
        utils.log_tabular("Train_Loss", -1)
        utils.log_tabular("Train_Score", -1)
        utils.log_tabular("Valid_Score", -1)
        utils.dump_tabular()

        return best_train_score, best_valid_score, np.mean(epoch_times)

    def arr_to_seq(self, arr):
        return decode_arr_to_seq(arr, self.idx2vocab)

    def predict(self, data_loader):
        self.net.eval()
        seq_ref_list = []
        seq_pred_list = []

        for step, batch in enumerate(data_loader):
            batch = batch.to(self.device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred_list = self.net(batch)

                mat = []
                for i in range(len(pred_list)):
                    mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
                mat = torch.cat(mat, dim=1)

                seq_pred = [self.arr_to_seq(arr) for arr in mat]
                seq_ref = [batch.y[i] for i in range(len(batch.y))]

                seq_ref_list.extend(seq_ref)
                seq_pred_list.extend(seq_pred)

        input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

        return self.evaluator.eval(input_dict)
