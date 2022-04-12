import os
import copy
import json
import time
import shutil
import numpy as np
from os.path import join

import torch
import torch.nn as nn
from torch.nn import functional as F

from Loss import Loss
from settings import get_args
from BasicTrainer import BasicTrainer


class MLC(BasicTrainer):

    def __init__(self, args):
        super().__init__(args)

        # Initialize Cooperation Models
        self.modelA = self._get_model(self.args.backbone)
        self.modelB = self._get_model(self.args.backbone)
        # Optimizer & Criterion
        self.loss = Loss(args)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(self.args.device)
        self.softmax = nn.Softmax(dim=1).to(self.args.device)
        self.optimizerA = self._get_optim(self.modelA.parameters(), optim=self.args.optim)
        self.optimizerB = self._get_optim(self.modelB.parameters(), optim=self.args.optim)
        self.rate_schedule = self._get_rate_schedule()
        # trainer init
        self._recoder_init()
        self._save_meta()
        # Load Data
        self.trainloader, self.testloader, self.valloader = self._load_data()
        # Optionally resume from a checkpoint
        if os.path.isfile(self.args.checkpoint_dir):
            self._resume()
        else:
            print("=> no checkpoint found at '{}'".format(self.args.checkpoint_dir))
            # save clean label & noisy label
            np.save(join(self.args.dir, 'y_clean.npy'), self.clean_labels)

    def _resume(self):
        # load model state
        print("=> loading checkpoint '{}'".format(self.args.checkpoint_dir))
        checkpoint = torch.load(self.args.checkpoint_dir)
        self.args.start_epoch = checkpoint['epoch']
        self.best_prec1 = checkpoint['best_prec1']
        self.modelA.load_state_dict(checkpoint['state_dict_A'])
        self.optimizerA.load_state_dict(checkpoint['optimizer_A'])
        self.modelB.load_state_dict(checkpoint['state_dict_B'])
        self.optimizerB.load_state_dict(checkpoint['optimizer_B'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.args.checkpoint_dir, checkpoint['epoch']))

        # load record_dict
        if os.path.isfile(self.args.record_dir):
            print("=> loading record file {}".format(self.args.record_dir))
            with open(self.args.record_dir, 'r') as f:
                self.record_dict = json.load(f)
                print("=> loaded record file {}".format(self.args.record_dir))

    def _recoder_init(self):
        os.makedirs(self.args.dir, exist_ok=True)
        os.makedirs(join(self.args.dir, 'record'), exist_ok=True)

        keys = ['acc', 'acc5', 'loss']
        record_infos = {}
        for k in keys:
            record_infos[k] = []
        # 3 is mixed model result
        self.record_dict = {
            'train1': copy.deepcopy(record_infos),
            'test1': copy.deepcopy(record_infos),
            'val1': copy.deepcopy(record_infos),
            'train2': copy.deepcopy(record_infos),
            'test2': copy.deepcopy(record_infos),
            'val2': copy.deepcopy(record_infos),
            'train3': copy.deepcopy(record_infos),
            'test3': copy.deepcopy(record_infos),
            'val3': copy.deepcopy(record_infos),
            'loss_val': [],
            'loss_avg': [],
        }

    def _record(self):
        # write file
        with open(self.args.record_dir, 'w') as f:
            json.dump(self.record_dict, f, indent=4, sort_keys=True)

    # define drop rate schedule
    def _get_rate_schedule(self):
        rate_schedule = np.ones(self.args.epochs) * self.args.forget_rate
        if self.args.warmup > 0:
            rate_schedule[:self.args.warmup] = 0
            rate_schedule[self.args.warmup:self.args.warmup+self.args.num_gradual] = np.linspace(self.args.warmup,
                                                                self.args.warmup + (
                                                                            self.args.forget_rate ** self.args.exponent),
                                                                self.args.num_gradual)
        else:
            rate_schedule[:self.args.num_gradual] = np.linspace(0,
                                                                self.args.forget_rate ** self.args.exponent,
                                                                self.args.num_gradual)
        return rate_schedule

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate"""
        if epoch < self.args.stage2:
            lr = self.args.lr
        elif epoch < (self.args.epochs - self.args.stage2) // 3 + self.args.stage2:
            lr = self.args.lr2
        elif epoch < 2 * (self.args.epochs - self.args.stage2) // 3 + self.args.stage2:
            lr = self.args.lr2 / 10
        else:
            lr = self.args.lr2 / 100
        for param_group in self.optimizerA.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizerB.param_groups:
            param_group['lr'] = lr

    def _compute_loss(self, outputA, outputB, target, target_var, index, epoch):
        if epoch < self.args.stage1:
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(
                target.size(0),
                self.args.classnum
            ).scatter_(1, target.view(-1, 1), self.args.K)
            onehot = onehot.numpy()
            self.new_y[index, :] = onehot
            # training as normal co-teaching
            forget_rate = self.rate_schedule[epoch]
            lossA, lossB = self.loss.loss_mlc(
                outputA, outputB,
                target_var, target_var,
                forget_rate,
                loss_type='CE',
                softmax=True,
                net_param1=self.modelA.parameters(),
                net_param2=self.modelB.parameters()
            )
            return lossA, lossB, onehot, onehot
        elif epoch < self.args.stage2:
            # using select data sample update parameters, other update label only
            yy_A = torch.tensor(self.yy[index, :], dtype=torch.float32,
                                requires_grad=True, device=self.args.device)
            yy_B = torch.tensor(self.yy[index, :], dtype=torch.float32,
                                requires_grad=True, device=self.args.device)
            # obtain label distributions (y_hat)
            last_y_var_A = self.softmax(yy_A)
            last_y_var_B = self.softmax(yy_B)
            # sort samples
            forget_rate = self.rate_schedule[epoch]
            lossA, lossB = self.loss.loss_mlc(
                outputA, outputB,
                last_y_var_A, last_y_var_B,
                forget_rate,
                loss_type="PENCIL",
                target_var=target_var,
                softmax=True,
                net_param1=self.modelA.parameters(),
                net_param2=self.modelB.parameters())
            return lossA, lossB, yy_A, yy_B
        else:
            yy_A = torch.tensor(self.yy[index, :], dtype=torch.float32,
                                requires_grad=True, device=self.args.device)
            yy_B = torch.tensor(self.yy[index, :], dtype=torch.float32,
                                requires_grad=True, device=self.args.device)
            last_y_var_A = self.softmax(yy_A)
            last_y_var_B = self.softmax(yy_B)
            forget_rate = self.rate_schedule[epoch]
            lossA, lossB = self.loss.loss_mlc(
                outputA, outputB,
                last_y_var_A,
                last_y_var_B,
                forget_rate,
                loss_type="FLIP_KL",
                softmax=True,
                net_param1=self.modelA.parameters(),
                net_param2=self.modelB.parameters()
            )
            return lossA, lossB, yy_A, yy_B

    def training(self):
        timer = AverageMeter()
        # train
        end = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            print('-----------------')

            self._adjust_learning_rate(epoch)

            # load y_tilde
            if epoch == self.args.start_epoch:
                if os.path.isfile(self.args.y_file):
                    self.yy = np.load(self.args.y_file)
                else:
                    self.yy = []

            train_prec1_A, train_prec1_B = self.train(epoch)
            val_prec1_A, val_prec1_B = self.val(epoch)
            test_prec1_A, test_prec1_B = self.test(epoch)

            is_best = max(val_prec1_A, val_prec1_B) > self.best_prec1
            self.best_prec1 = max(val_prec1_A, val_prec1_B, self.best_prec1)
            if self.args.save_ckpt:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': self.args.backbone,
                        'state_dict_A': self.modelA.state_dict(),
                        'optimizer_A': self.optimizerA.state_dict(),
                        'state_dict_B': self.modelB.state_dict(),
                        'optimizer_B': self.optimizerB.state_dict(),
                        'prec_A': val_prec1_A,
                        'prec_B': val_prec1_B,
                        'best_prec1': self.best_prec1,
                    },
                    is_best,
                    filename=self.args.checkpoint_dir,
                    modelbest=self.args.modelbest_dir
                )

            if self.args.record_in_train:
                self._record()

            timer.update(time.time() - end)
            end = time.time()
            print("Epoch {} using {} min {:.2f} sec".format(epoch, timer.val // 60, timer.val % 60))
        self._record()

    def train(self, epoch=0):
        batch_time = AverageMeter()
        losses_A = AverageMeter()
        losses_B = AverageMeter()
        top1_A = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        end = time.time()

        self.modelA.train()
        self.modelB.train()

        # new y is y_tilde after updating
        if epoch >= self.args.stage1:
            self.new_y = self.yy
        else:
            self.new_y = np.zeros([self.args.datanum, self.args.classnum])

        for i, (input, target, index) in enumerate(self.trainloader):

            index = index.numpy()
            input = input.to(self.args.device)
            target1 = target.to(self.args.device)
            input_var = input.clone().to(self.args.device)
            target_var = target1.clone().to(self.args.device)

            outputA = self.modelA(input_var)
            outputB = self.modelB(input_var)
            output_mix = (outputA + outputB) / 2

            if epoch < self.args.warmup:
                lossA = self.loss.get_loss(outputA, target1, loss_type="CE")
                lossB = self.loss.get_loss(outputB, target1, loss_type="CE")
            else:
                lossA, lossB, yy_A, yy_B = self._compute_loss(
                    outputA, outputB, target, target_var, index, epoch
                )

            outputA_ = outputA
            outputB_ = outputB

            outputA = F.softmax(outputA, dim=1)
            outputB = F.softmax(outputB, dim=1)
            output_mix = F.softmax(output_mix, dim=1)

            # Update recorder
            prec1_A = accuracy(outputA.data, target1, topk=(1,))
            prec1_B = accuracy(outputB.data, target1, topk=(1,))
            prec1_mix = accuracy(output_mix.data, target1, topk=(1,))
            top1_A.update(float(prec1_A[0]), input.shape[0])
            top1_B.update(float(prec1_B[0]), input.shape[0])
            top1_mix.update(float(prec1_mix[0]), input.shape[0])
            losses_A.update(float(lossA.data))
            losses_B.update(float(lossA.data))

            self.optimizerA.zero_grad()
            lossA.backward(retain_graph=True)
            self.optimizerB.zero_grad()
            lossB.backward()
            self.optimizerA.step()
            self.optimizerB.step()

            # update label distribution
            if epoch >= self.args.stage1 and epoch < self.args.stage2:
                # using select data sample update parameters, other update label only
                yy_A = torch.tensor(self.yy[index, :], dtype=torch.float32,
                                    requires_grad=True, device=self.args.device)
                yy_B = torch.tensor(self.yy[index, :], dtype=torch.float32,
                                    requires_grad=True, device=self.args.device)
                # obtain label distributions (y_hat)
                last_y_var_A = self.softmax(yy_A)
                last_y_var_B = self.softmax(yy_B)
                # re-compute loss
                lossA = self.loss.get_loss(outputA_.detach(), last_y_var_A,
                                           loss_type="PENCIL", target_var=target_var)
                lossB = self.loss.get_loss(outputB_.detach(), last_y_var_B,
                                           loss_type="PENCIL", target_var=target_var)

                lossA.backward()
                lossB.backward()

                # update y_tile
                grad = yy_A.grad.data + yy_B.grad.data
                yy_A.data.sub_(self.args.lambda1 * grad)
                self.new_y[index, :] = yy_A.data.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print("\rTrain Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                  "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                  "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                  "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                  "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                epoch, self.args.epochs, i, self.train_batch_num,
                batch_time=batch_time, loss_A=losses_A, loss_B=losses_B,
                top1_A=top1_A, top1_B=top1_B),
                end=''
            )

        if epoch < self.args.stage2:
            # save y_tilde
            self.yy = self.new_y
            y_file = join(self.args.dir, "y.npy")
            if (epoch+1) == self.args.stage2:
                np.save(y_file, self.new_y)
            if args.save_all_y or epoch == 0:
                y_record = join(self.args.dir, "record/y_%03d.npy" % epoch)
                np.save(y_record, self.new_y)

        print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['train1']['acc'].append(top1_A.avg)
        self.record_dict['train1']['loss'].append(losses_A.avg)
        self.record_dict['train2']['acc'].append(top1_B.avg)
        self.record_dict['train2']['loss'].append(losses_B.avg)
        self.record_dict['train3']['acc'].append(top1_mix.avg)

        return top1_A.avg, top1_B.avg


    def val(self, epoch=0):
        self.modelA.eval()
        self.modelB.eval()

        batch_time = AverageMeter()
        losses_A = AverageMeter()
        top1_A = AverageMeter()
        losses_B = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()

        with torch.no_grad():
            # Validate
            end = time.time()
            for i, (img, label, index) in enumerate(self.valloader):

                img = img.to(self.args.device)
                label = label.to(self.args.device)

                outputA = self.modelA(img)
                lossA = self.criterion(outputA, label)
                outputB = self.modelB(img)
                lossB = self.criterion(outputB, label)
                output_mix = (outputA + outputB) / 2

                outputA = F.softmax(outputA, dim=1)
                outputB = F.softmax(outputB, dim=1)
                output_mix = F.softmax(output_mix, dim=1)

                # Update recorder
                prec1_A = accuracy(outputA.data, label, topk=(1,))
                prec1_B = accuracy(outputB.data, label, topk=(1,))
                prec1_mix = accuracy(output_mix.data, label, topk=(1,))
                top1_A.update(float(prec1_A[0]), img.shape[0])
                losses_A.update(float(lossA.data))
                top1_B.update(float(prec1_B[0]), img.shape[0])
                losses_B.update(float(lossB.data))
                top1_mix.update(float(prec1_mix[0]), img.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print("\rVal Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                      "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                      "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                      "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                      "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                    epoch, self.args.epochs, i, self.val_batch_num,
                    batch_time=batch_time, loss_A=losses_A, loss_B=losses_B,
                    top1_A=top1_A, top1_B=top1_B), end='')

            print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['val1']['acc'].append(top1_A.avg)
        self.record_dict['val1']['loss'].append(losses_A.avg)
        self.record_dict['val2']['acc'].append(top1_B.avg)
        self.record_dict['val2']['loss'].append(losses_B.avg)
        self.record_dict['val3']['acc'].append(top1_mix.avg)

        return top1_A.avg, top1_B.avg

    def test(self, epoch=0):
        self.modelA.eval()
        self.modelB.eval()

        batch_time = AverageMeter()
        losses_A = AverageMeter()
        top1_A = AverageMeter()
        losses_B = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()

        with torch.no_grad():
            # Validate
            end = time.time()
            for i, (img, label, index) in enumerate(self.testloader):

                img = img.to(self.args.device)
                label = label.to(self.args.device)

                outputA = self.modelA(img)
                lossA = self.criterion(outputA, label)
                outputB = self.modelB(img)
                lossB = self.criterion(outputB, label)
                output_mix = (outputA + outputB) / 2

                outputA = F.softmax(outputA, dim=1)
                outputB = F.softmax(outputB, dim=1)
                output_mix = F.softmax(output_mix, dim=1)

                # Update recorder
                prec1_A = accuracy(outputA.data, label, topk=(1,))
                prec1_B = accuracy(outputB.data, label, topk=(1,))
                prec1_mix = accuracy(output_mix.data, label, topk=(1,))
                top1_A.update(float(prec1_A[0]), img.shape[0])
                losses_A.update(float(lossA.data))
                top1_B.update(float(prec1_B[0]), img.shape[0])
                losses_B.update(float(lossB.data))
                top1_mix.update(float(prec1_mix[0]), img.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print("\rTest Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                      "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                      "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                      "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                      "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                    epoch, self.args.epochs, i, self.test_batch_num,
                    batch_time=batch_time, loss_A=losses_A, loss_B=losses_B,
                    top1_A=top1_A, top1_B=top1_B), end='')

            print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['test1']['acc'].append(top1_A.avg)
        self.record_dict['test1']['loss'].append(losses_A.avg)
        self.record_dict['test2']['acc'].append(top1_B.avg)
        self.record_dict['test2']['loss'].append(losses_B.avg)
        self.record_dict['test3']['acc'].append(top1_mix.avg)

        return top1_A.avg, top1_B.avg

def save_checkpoint(state, is_best, filename='', modelbest=''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)
        print("Saving best model at epoch {}".format(state['epoch']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    args = get_args()
    trainer = MLC(args=args)
    trainer.training()
