import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

epoch_list, nfe_forward_list, nfe_backward_list = [], [], []
train_acc_list, test_acc_list, time_list = [], [], []

result_dir = './tab1/odenet'
save_dir = './fig3'
os.makedirs(save_dir, exist_ok=True)

with open(f'{result_dir}/logs.txt', 'r') as f:
    for line in f:
        match = re.match(r"Epoch (\d+) \| Time ([\d.]+) \([\d.]+\) \| NFE-F ([\d.]+) \| NFE-B ([\d.]+) \| Train Acc ([\d.]+) \| Test Acc ([\d.]+)", line)
        if match:
            epoch = int(match.group(1))
            time = float(match.group(2))
            nfe_forward = float(match.group(3))
            nfe_backward = float(match.group(4))
            train_acc = float(match.group(5))
            test_acc = float(match.group(6))

            epoch_list.append(epoch)
            time_list.append(time)
            nfe_forward_list.append(nfe_forward)
            nfe_backward_list.append(nfe_backward)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

numerical_error_list = [1 - acc for acc in train_acc_list]
plt.figure(figsize=(5, 5))
plt.scatter(nfe_forward_list, numerical_error_list, c=numerical_error_list, cmap='rainbow', s=800, norm=LogNorm())
plt.colorbar(label='Numerical Error (Log Scale)')
plt.xlabel('NFE Forward')
plt.ylabel('Numerical Error (1 - Train Acc)')
plt.title('(a) NFE Forward vs Numerical Error')
plt.savefig(f'{save_dir}/a.png', dpi=300)
plt.show()

relative_time_list = [t / max(time_list) for t in time_list]
plt.figure(figsize=(5, 5))
plt.scatter(nfe_forward_list, relative_time_list, c=relative_time_list, cmap='rainbow', s=800, norm=LogNorm())
plt.colorbar(label='Relative Time (Log Scale)')
plt.xlabel('NFE Forward')
plt.ylabel('Relative Time')
plt.title('(b) NFE Forward vs Relative Time')
plt.savefig(f'{save_dir}/b.png', dpi=300)
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(nfe_forward_list, nfe_backward_list, c=nfe_backward_list, cmap='rainbow', s=800, norm=LogNorm())
plt.colorbar(label='NFE Backward (Log Scale)')
plt.xlabel('NFE Forward')
plt.ylabel('NFE Backward')
plt.title('(c) NFE Forward vs NFE Backward')
plt.savefig(f'{save_dir}/c.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epoch_list, nfe_forward_list, label='NFE Forward')
plt.xlabel('Epoch')
plt.ylabel('NFE Forward')
plt.title('(d) Training Epoch vs NFE Forward')
plt.grid(True)
plt.savefig(f'{save_dir}/d.png', dpi=300)
plt.show()