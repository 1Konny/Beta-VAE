"""solver.py"""

import time
from pathlib import Path

import visdom
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid

from utils import cuda
from model import BetaVAE
from dataset import return_data


def original_vae_loss(x, x_recon, mu, logvar):
    batch_size = x.size(0)
    if batch_size == 0:
        recon_loss = 0
        kl_divergence = 0
    else:
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
        # kld which one is correct? from the equation in most of papers,
        # I think the first one is correct but official pytorch code uses the second one.

        kl_divergence = -0.5*(1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
        #kl_divergence = -0.5*(1 + logvar - mu**2 - logvar.exp()).sum()
        #dimension_wise_kl_divergence = -0.5*(1 + logvar - mu**2 - logvar.exp()).mean(0)

    return recon_loss, kl_divergence


class Solver(object):
    def __init__(self, args):

        # Misc
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.beta = args.beta

        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.net = cuda(BetaVAE(self.z_dim), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        # Visdom
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        if self.viz_on:
            self.viz = visdom.Visdom(env=self.viz_name, port=self.viz_port)
            self.viz_curves = visdom.Visdom(env=self.viz_name+'/train_curves', port=self.viz_port)
            self.win_recon = None
            self.win_kld = None

        # Checkpoint
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.viz_name)
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.load_ckpt = args.load_ckpt
        if self.load_ckpt:
            self.load_checkpoint()

        # Data
        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

    def train(self):
        self.net_mode(train=True)
        out = False
        while not out:
            start = time.time()
            curve_data = []
            for x in self.data_loader:
                self.global_iter += 1

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss, kld = original_vae_loss(x, x_recon, mu, logvar)

                beta_vae_loss = recon_loss + self.beta*kld

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.global_iter%1000 == 0:
                    curve_data.append(torch.Tensor([self.global_iter,
                                                    recon_loss.data[0],
                                                    kld.data[0],]))

                if self.global_iter%5000 == 0:
                    self.save_checkpoint()
                    self.visualize(dict(image=[x, x_recon], curve=curve_data))
                    print('[{}] recon_loss:{:.3f} beta*kld:{:.3f}'.format(
                        self.global_iter, recon_loss.data[0], self.beta*kld.data[0]))
                    curve_data = []

                if self.global_iter >= self.max_iter:
                    out = True
                    break

            end = time.time()
            print('[time elapsed] {:.2f}s/epoch'.format(end-start))
        print("[Training Finished]")

    def visualize(self, data):
        x, x_recon = data['image']
        curve_data = data['curve']

        sample_x = make_grid(x.data.cpu(), normalize=False)
        sample_x_recon = make_grid(F.sigmoid(x_recon).data.cpu(), normalize=False)
        samples = torch.stack([sample_x, sample_x_recon], dim=0)
        self.viz.images(samples, opts=dict(title=str(self.global_iter)))

        curve_data = torch.stack(curve_data, dim=0)
        curve_iter = curve_data[:, 0]
        curve_recon = curve_data[:, 1]
        curve_kld = curve_data[:, 2]

        if self.win_recon is None:
            self.win_recon = self.viz_curves.line(
                                        X=curve_iter,
                                        Y=curve_recon,
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='reconsturction loss',))
        else:
            self.win_recon = self.viz_curves.line(
                                        X=curve_iter,
                                        Y=curve_recon,
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='reconsturction loss',))

        if self.win_kld is None:
            self.win_kld = self.viz_curves.line(
                                        X=curve_iter,
                                        Y=curve_kld,
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='kl divergence',))
        else:
            self.win_kld = self.viz_curves.line(
                                        X=curve_iter,
                                        Y=curve_kld,
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            xlabel='iteration',
                                            ylabel='kl divergence',))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename='ckpt.tar', silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
