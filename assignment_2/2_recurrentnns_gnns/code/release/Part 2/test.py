import random
import torch
import os
import string

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# dataset = dataset.TextDataset("{}/assets/book_EN_grimms_fairy_tails.txt".÷format(BASE_DIR), 50)  # fixme
weights = torch.rand([1, 5,3]) * 10
weights = torch.nn.functional.softmax(weights, dim = 2)
print(weights)
print("1", torch.multinomial(torch.squeeze(weights), 1).view(1,-1))
print("2", torch.distributions.categorical.Categorical(weights).sample())
print(torch.argmax(weights,dim=2))