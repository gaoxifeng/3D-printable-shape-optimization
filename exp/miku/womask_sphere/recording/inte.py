import torch
from torch import nn
import torch.nn.functional as F
import time
# input = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2)
# print(f'{input},\n')
input = torch.randn((1, 4, 256, 512, 1024))
grid = torch.randn((1, 1, 1, 100, 3))
t_start = time.time()
output = F.grid_sample(input,grid,mode='bilinear',align_corners=True)
print(output.shape)
print('time per iteration ', time.time() - t_start)