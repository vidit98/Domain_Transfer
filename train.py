import torch.nn as nn
import numpy as np
import argparse
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module

from model import Model
from loss import loss, AverageMeter


from Generator import HandWrtng
from Discriminators import Discr
from FTN import FTNet

from data import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_ckpt(loss, net, epoch):
	torch.save(net.state_dict(), 'epoch{}_{}'.format(epoch, args.num_epoch))
	f = open("losses.txt", "a")
	f.write('Epoch{}, Loss F:{}, Loss G:{}, Loss D1:{}, Loss D2:{}'.format(epoch, loss[0], loss[1], loss[2], loss[3]))
	f.write("\n")
	f.close()

def train(model, data, optimizer, epoch, args):
	lf, lg, lD1, lD2 = (0,0,0,0)
	avg_losslf = AverageMeter()
	avg_losslg = AverageMeter()
	avg_lossld1 = AverageMeter()
	avg_lossld2 = AverageMeter()
	for i in range(args.epoch_iters):

		train_data = data[i]
		out = model(train_data[0].to(device), train_data[1].to(device), train_data[2].to(device))

		optimizer.zero_grad()

		lf, lg, ld1, ld2 = loss(out[0], out[1], out[2], out[3], out[4], train_data[3].to(device), out[5], out[6], out[7], out[8], out[9])

		total_norm = 0
		for p in model.parameters():
			total_norm += torch.sum(torch.abs(p))
		total_norm = total_norm ** (1. / 2)

		if i%2:
			lf.backward()
			lg.backward()
			ld2.backward()
		else:
			ld1.backward()

		optimizer.step()

		avg_losslf.update(lf)
		avg_losslg.update(lg)
		avg_lossld1.update(ld1)
		avg_lossld2.update(ld2)

		if i % args.disp_iter == 0:
			print('Epoch: [{}][{}/{}],'
                  'lr: {:.6f}, '
                  ' LossF: {:.6f}, LossG: {:.6f},LossD1: {:.6f},LossD2: {:.6f}, Grads: {:.6f}'
                  .format(epoch, i, args.epoch_iters,
                          args.lr,
                          avg_losslf.avg, avg_losslg.avg, avg_lossld1.avg, avg_lossld2.avg , total_norm))
	save_ckpt((lf, lg, lD1, lD2), model, epoch)
	



def create_optimizer(net, args):

	optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
	return optimizer

def main(args):
	model = Model().to(device)
	if args.load:
		model.load_state_dict(torch.load("epoch12_20"))
	mnist = MNIST(args.list_domainS, args.list_domainT, args.list_root)
	optimizer = create_optimizer(model, args)
	# xs = torch.randn(5, 3, 32 ,32)
	# xsp = torch.randn(5,3,32,32)
	# xt = torch.randn(5,3,32,32)

	# r = model(xs, xsp, xt)
	# t = torch.from_numpy(np.array([0,1,2,3,4]))

	# l = loss(r[0], r[1], r[2], r[3], r[4], t, r[5], r[6], r[7], r[8], r[9])

	for epoch in range(args.start_epoch, args.num_epoch + 1):
		train(model, mnist, optimizer, epoch, args)

	




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--list_domainS',
                        default='mnist_m/mnist_m_train_labels.txt')
    parser.add_argument('--list_domainT',
                        default='mnist.txt')
    parser.add_argument('--load',
                        default=1)
    parser.add_argument('--list_root',
                        default='mnist_m/mnist_m_train')
    parser.add_argument('--gpus', default='0,1,2',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=20, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=6000, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='SGD', help='optimizer')
    parser.add_argument('--lr', default=1e-5, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--disp_iter', type=int, default=50,
                        help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    main(args)