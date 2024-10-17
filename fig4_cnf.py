import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from torchdiffeq import odeint
from PIL import Image

class CNF(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            W, B, U = self.hyper_net(t)
            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)
            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
        return (dz_dt, dlogp_z_dt)

class HyperNetwork(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        blocksize = width * in_out_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)
        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]

def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def get_batch(num_samples, device):
    theta = torch.linspace(0, np.pi, num_samples // 2)

    x1 = torch.sin(theta) * 0.6 + 0.8
    y1 = torch.cos(theta) 
    curve1 = torch.stack([x1, y1], dim=1)

    x2 = -torch.sin(theta) * 0.6 - 0.8
    y2 = torch.cos(theta)
    curve2 = torch.stack([x2, y2], dim=1)

    data = torch.cat([curve1, curve2], dim=0)
    noise = torch.randn_like(data) * 0.1
    data += noise

    x = data.to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    
    return (x, logp_diff_t1)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
in_out_dim = 2
hidden_dim = 32
width = 2
niters = 10000
lr = 1e-3
num_samples = 512
results_dir = f"./fig4a/cnf_{width}"
os.makedirs(results_dir, exist_ok=True)
func = CNF(in_out_dim=in_out_dim, hidden_dim=hidden_dim, width=width).to(device)
optimizer = torch.optim.Adam(func.parameters(), lr=lr)

p_z0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0]).to(device),
    covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
)

for itr in range(1, niters + 1):
    optimizer.zero_grad()
    x, logp_diff_t1 = get_batch(num_samples, device)
    t0, t1 = 0, 10

    z_t, logp_diff_t = odeint(
        func, (x, logp_diff_t1), 
        torch.tensor([t1, t0]).type(torch.float32).to(device), 
        atol=1e-5, rtol=1e-5
    )
    z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

    p_log_prob = p_z0.log_prob(z_t0)
    q_log_prob = -logp_diff_t0.view(-1)
    kld_loss = (q_log_prob - p_log_prob).mean()

    kld_loss.backward()
    optimizer.step()

    if itr % 100 == 0:
        print(f"Iter {itr}, KLD Loss: {kld_loss.item():.4f}")

viz_samples = 30000
viz_timesteps = 41
target_sample, _ = get_batch(viz_samples, device)

with torch.no_grad():
    z_t0 = p_z0.sample([viz_samples]).to(device)
    logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

    z_t_samples, _ = odeint(
        func, (z_t0, logp_diff_t0),
        torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
        atol=1e-5, rtol=1e-5
    )

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

    z_t1 = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

    z_t_density, logp_diff_t = odeint(
        func, (z_t1, logp_diff_t1),
        torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
        atol=1e-5, rtol=1e-5
    )

    for (t, z_sample, z_density, logp_diff) in zip(
            np.linspace(t0, t1, viz_timesteps), z_t_samples, z_t_density, logp_diff_t
    ):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        plt.tight_layout()
        plt.axis('off')
        fig.suptitle(f'{t:.2f}s')
        plt.margins(0, 0)

        plt.subplot(1, 3, 1)
        plt.hist2d(*target_sample.cpu().numpy().T, bins=300, density=True, range=[[-3, 3], [-3, 3]])

        plt.subplot(1, 3, 2)
        plt.hist2d(*z_sample.cpu().numpy().T, bins=300, density=True, range=[[-3, 3], [-3, 3]])

        plt.subplot(1, 3, 3)
        logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
        plt.tricontourf(*z_t1.cpu().numpy().T, np.exp(logp.cpu().numpy()), 200)

        plt.savefig(os.path.join(results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"), bbox_inches='tight')
        plt.close()

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(results_dir, "cnf-viz-*.jpg")))]
    img.save(fp=os.path.join(results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs, save_all=True, duration=250, loop=0)