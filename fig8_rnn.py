import os
import argparse
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--train_dir', type=str, default='./fig8/rnn')
args = parser.parse_args()

def generate_spiral2d(nspiral=1000, ntotal=500, nsample=100, start=0., stop=1, noise_std=.1, a=0., b=1.):
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    plt.figure()
    plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
    plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
    plt.legend()
    plt.savefig('./ground_truth.png', dpi=500)
    print('Saved ground truth spiral at ./ground_truth.png')

    orig_trajs, samp_trajs = [], []
    for _ in range(nspiral):
        t0_idx = np.argmax(npr.multinomial(1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))) + nsample
        cc = bool(npr.rand() > .5)
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    return np.stack(orig_trajs, axis=0), np.stack(samp_trajs, axis=0), orig_ts, samp_ts

class LatentRNN(nn.Module):
    def __init__(self, input_dim=2, latent_dim=4, nhidden=20, n_layers=1):
        super(LatentRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, latent_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(latent_dim, input_dim)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, latent_dim).to(device)

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def generate_trajectory_rnn(model, initial_obs, hidden, steps, forward=True):
    traj = [initial_obs]
    current_input = initial_obs

    for _ in range(steps - 1):
        output, hidden = model(current_input, hidden)
        traj.append(output)
        
        current_input = output if forward else current_input

    return torch.cat(traj, dim=1)

if __name__ == '__main__':
    input_dim, latent_dim, nhidden, n_layers = 2, 4, 20, 1
    nspiral, start, stop, noise_std = 1000, 0., 6 * np.pi, 0.3
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral, start=start, stop=stop, noise_std=noise_std, a=0., b=0.3
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)

    rnn = LatentRNN(input_dim=input_dim, latent_dim=latent_dim, nhidden=nhidden).to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
    loss_meter = RunningAverageMeter()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        hidden = rnn.init_hidden(samp_trajs.size(0), device)

        for t in range(samp_trajs.size(1)):
            obs = samp_trajs[:, t:t+1, :]
            output, hidden = rnn(obs, hidden)

        loss = F.mse_loss(output, samp_trajs[:, -1:, :])
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        if itr % 100 == 0:
            print(f'Iter: {itr}, Running Avg Loss: {loss_meter.avg:.4f}')

    print('Training complete.')

    with torch.no_grad():
        hidden_forward = rnn.init_hidden(1, device)
        hidden_backward = rnn.init_hidden(1, device)

        initial_obs_forward = samp_trajs[0:1, 0:1, :].to(device)
        pos_traj = generate_trajectory_rnn(rnn, initial_obs_forward, hidden_forward, steps=200, forward=True)

        initial_obs_backward = samp_trajs[0:1, -1:, :].to(device)
        neg_traj = generate_trajectory_rnn(rnn, initial_obs_backward, hidden_backward, steps=200, forward=False)

        xs_pos = pos_traj.cpu().numpy()
        xs_neg = neg_traj.cpu().numpy()
        orig_traj = orig_trajs[0].cpu().numpy()
        samp_trajs = samp_trajs[0].cpu().numpy()

        plt.figure()
        plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true trajectory')
        plt.plot(xs_pos[0, :, 0], xs_pos[0, :, 1], 'r', label='learned trajectory (t>0)')
        plt.plot(xs_neg[0, :, 0], xs_neg[0, :, 1], 'c', label='learned trajectory (t<0)')
        plt.scatter(samp_trajs[:, 0], samp_trajs[:, 1], label='sampled data', s=3)

        plt.legend(loc='best')
        plt.savefig('./rnn_vis.png', dpi=500)
        print('Saved RNN visualization at ./rnn_vis.png')