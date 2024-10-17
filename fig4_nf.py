import torch
import numpy as np
import normflows as nf
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

K = 2
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
results_dir = f"./fig4a/nf_{K}"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

flows = []
for i in range(K):
    flows += [nf.flows.Planar((2,))]
target = nf.distributions.TwoModes(2, 0.1)

q0 = nf.distributions.DiagGaussian(2)
nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
nfm.to(device)

grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
z = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
log_prob = target.log_prob(z.to(device)).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)

plt.figure(figsize=(10, 10))
plt.pcolormesh(xx, yy, prob)
plt.savefig(os.path.join(results_dir, 'target.png'))

z, _ = nfm.sample(num_samples=2 ** 20)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
plt.savefig(os.path.join(results_dir, 'init_flow.png'))

max_iter = 20000
num_samples = 40
anneal_iter = 10000
show_iter = 2000

loss_hist = np.array([])

optimizer = torch.optim.RMSprop(nfm.parameters(), lr=1e-3, weight_decay=1e-4)
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    loss = nfm.reverse_kld(num_samples, beta=np.min([1., 0.01 + it / anneal_iter]))
    loss.backward()
    optimizer.step()
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    if (it + 1) % show_iter == 0:
        torch.cuda.manual_seed(0)
        z, _ = nfm.sample(num_samples=2 ** 20)
        z_np = z.to('cpu').data.numpy()
        
        plt.figure(figsize=(10, 10))
        plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
        plt.savefig(os.path.join(results_dir, f'learned_dist_{it+1}.png'))

plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss.png'))
print(f'Final loss: {loss_hist[-1]}')

z, _ = nfm.sample(num_samples=2 ** 20)
z_np = z.to('cpu').data.numpy()
plt.figure(figsize=(10, 10))
plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (grid_size, grid_size), range=[[-3, 3], [-3, 3]])
plt.savefig(os.path.join(results_dir, 'final_dist.png'))