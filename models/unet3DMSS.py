# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F
import torchcomplex.nn.functional as cF

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class UNetMSS(nn.Module):
    """
    Implementation of
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)
    https://arxiv.org/abs/1505.04597

    Using the default arguments will yield the exact version used
    in the original paper

    Args:
        in_channels (int): number of input channels
        n_classes (int): number of output channels
        depth (int): depth of the network
        wf (int): number of filters in the first layer is 2**wf
        padding (bool): if True, apply padding such that the input shape
                        is the same as the output.
                        This may introduce artifacts
        batch_norm (bool): Use BatchNorm after layers with an
                            activation function
        up_mode (str): one of 'upconv' or 'upsample'.
                        'upconv' will use transposed convolutions for
                        learned upsampling.
                        'upsample' will use bilinear upsampling.
    """
    def __init__(self, in_channels=1, n_classes=1, depth=3, wf=6, padding=True,
                 batch_norm=False, up_mode='upconv', dropout=False, mss_level=2, mss_fromlatent=True, 
                 mss_up="trilinear", mss_interpb4=False):
        super(UNetMSS, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.dropout = nn.Dropout3d() if dropout else nn.Sequential()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        up_out_features = []
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        if mss_fromlatent:
            mss_features = [prev_channels]
        else:
            mss_features = []

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)
            up_out_features.append(prev_channels)

        self.last = nn.Conv3d(prev_channels, n_classes, kernel_size=1)

        mss_features += up_out_features[len(up_out_features)-1-mss_level if not mss_fromlatent 
                                                                        else len(up_out_features)-1-mss_level+1:-1]

        self.mss_level = mss_level
        self.mss_up = mss_up
        self.mss_fromlatent = mss_fromlatent
        self.mss_interpb4 = mss_interpb4
        self.mss_convs = nn.ModuleList()
        for i in range(self.mss_level):
            self.mss_convs.append(nn.Conv3d(mss_features[i], n_classes, kernel_size=1))
        if self.mss_level == 1:
            self.mss_coeff = [0.5]
        else:
            lmbda = []
            for i in range(self.mss_level-1, -1, -1):
                lmbda.append(2**i)
            self.mss_coeff = []
            fact = 1.0 / sum(lmbda)
            for i in range(self.mss_level-1):
                self.mss_coeff.append(fact*lmbda[i])
            self.mss_coeff.append(1.0 - sum(self.mss_coeff))          
            self.mss_coeff.reverse()


    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool3d(x, 2)
        x = self.dropout(x)
            
        if self.mss_fromlatent:
            mss = [x]
        else:
            mss = []

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
            if self.training and ((len(self.up_path)-1-i <= self.mss_level) and not(i+1 == len(self.up_path))):
                mss.append(x)

        if self.training:
            for i in range(len(mss)):
                if not self.mss_interpb4:
                    mss[i] = F.interpolate(self.mss_convs[i](mss[i]), size=x.shape[2:], mode=self.mss_up) 
                else:
                    mss[i] = self.mss_convs[i](F.interpolate(mss[i], size=x.shape[2:], mode=self.mss_up)) 
            
            return self.last(x), mss
        else:
            return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv3d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='trilinear', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_z = (layer_depth - target_size[0]) // 2
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        return layer[:, :, diff_z:(diff_z + target_size[0]), diff_y:(diff_y + target_size[1]), diff_x:(diff_x + target_size[2])]
        #  _, _, layer_height, layer_width = layer.size() #for 2D data
        # diff_y = (layer_height - target_size[0]) // 2
        # diff_x = (layer_width - target_size[1]) // 2
        # return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        # bridge = self.center_crop(bridge, up.shape[2:]) #sending shape ignoring 2 digit, so target size start with 0,1,2
        up = F.interpolate(up, size=bridge.shape[2:], mode='trilinear')
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out