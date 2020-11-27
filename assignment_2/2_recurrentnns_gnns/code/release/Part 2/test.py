import random
import torch
# torch.LongTensor
x = random.sample(range(0,4),3)
print(x)
x = torch.LongTensor(x)

print(x.view(1, -1).shape)