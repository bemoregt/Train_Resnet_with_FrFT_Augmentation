import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

import urllib.request
import tarfile
import os

url = 'https://download.pytorch.org/tutorial/hymenoptera_data.tar.gz'
filename = 'hymenoptera_data.tar.gz'

if not os.path.exists('hymenoptera_data'):
    urllib.request.urlretrieve(url, filename)
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    print("Data downloaded and extracted")
else:
    print("Data already exists")
    
    