# coding:utf-8
import torch, time, copy
import numpy as np
import torch.nn as nn
from net import Net
import utils
from tqdm import tqdm
from ogb.graphproppred import Evaluator


class MatPool():
    def __init__(self, args):
        self.args = args
        self.print_freq = args.print_freq
        self.device = args.device
        self.net = Net(args).to(self.device)
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

        num_params = sum(p.numel() for p in self.net.parameters())
        print(f'#Params: {num_params}')

        # self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.cls_criterion = torch.nn.BCEWithLogitsLoss()
        # self.cls_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100]), reduction='sum').to(self.device)
        self.reg_criterion = torch.nn.MSELoss()
        self.evaluator = Evaluator(args.dataset)

    def fit(self, train_loader, valid_loader, epochs, task_type):
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
            y_true = []
            y_pred = []
            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                batch = batch.to(self.device)
                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    pass
                else:
                    pred = self.net(batch)
                    self.optimizer.zero_grad()
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    if "classification" in task_type:
                        loss = self.cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    else:
                        loss = self.reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    loss.backward()
                    self.optimizer.step()

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
                losses.append(loss.item())

            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            # validation process
            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            train_score = self.evaluator.eval(input_dict)['ap']
            valid_score = self.predict(valid_loader)['ap']

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

    def predict(self, data_loader):
        self.net.eval()
        y_true = []
        y_pred = []
        for step, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = self.net(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return self.evaluator.eval(input_dict)
