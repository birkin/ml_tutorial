import logging

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models


# import torchvision  # must be imported before torch to avoid circular import error  ¯\_(ツ)_/¯
# import torch
# import torch.nn.functional as F
# from torch import nn


## setup logging ----------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger(__name__)
log.debug( 'logging ready' )


## hyper parameters
epochs = 10
batch_size = 32
learning_rate = 0.0001


## for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.debug( f'device: {device}' )


## Prepare the dataset
transform_train = transforms.Compose(
    [   transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
log.debug( f'transform_train, ``{transform_train}``') 
"""
log statement yields...
transform_train, ``Compose(
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=warn)
    RandomHorizontalFlip(p=0.5)
    RandomAffine(degrees=[0.0, 0.0], scale=(0.8, 1.2), shear=[-10.0, 10.0])
    ColorJitter(brightness=(0.0, 2.0), contrast=(0.0, 2.0), saturation=(0.0, 2.0), hue=None)
    ToTensor()
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
)``
"""
transform = transforms.Compose(
    [   transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
log.debug( f'transform, ``{transform}``' )
"""
Yields...
transform, ``Compose(
    Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=warn)
    ToTensor()
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
)``
"""


## next...
train_dataset = datasets.ImageFolder('../flower_photos/train', transform=transform_train)
val_dataset = datasets.ImageFolder('../flower_photos/test', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

log.debug( f'train_dataset, ``{train_dataset}``' )
log.debug( f'val_dataset, ``{val_dataset}``' )
