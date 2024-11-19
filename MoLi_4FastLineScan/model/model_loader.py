import torch
import torch.nn as nn
import os
from collections import OrderedDict
from model.CTCNet import Deep_Image_Prior_Network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def CTC_model_load(rank):
    im_net = Deep_Image_Prior_Network(rank, 'reflection',
                            upsample_mode=['nearest', 'nearest', 'bilinear'],
                            skip_n33d=32,
                            skip_n33u=32,
                            skip_n11=12,
                            num_scales=3,
                            n_channels=119)                                             
    return [im_net.to(device)]

