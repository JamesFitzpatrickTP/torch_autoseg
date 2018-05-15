import torch
import torch.nn as nn


def conv_conv(filt_size, in_filt, out_filt):
    return nn.Sequential(
        nn.Conv2d(in_filt, out_filt, filt_size, 1, 1),
        nn.ReLU(),
        nn.Conv2d(out_filt, out_filt, filt_size, 1, 1),
        nn.ReLU())
