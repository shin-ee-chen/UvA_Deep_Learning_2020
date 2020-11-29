import random
import torch
import os


# torch.LongTensor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TXT_FILE = '{}/assets/book_EN_grimms_fairy_tails.txt'.format(BASE_DIR)

print(os.path.basename(DEFAULT_TXT_FILE))