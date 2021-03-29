import math
import torch.nn as nn
from models import *

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class ThisNewNet(nn.Module):
    def __init__(self, scale_factor, loss_func=None, in_channels=1, n_classes=1, depth=3, batch_norm=False, up_mode="upsample", dropout=0.0, num_features=64, sliceup_first=False, loss_slice_count=2, loss_inplane=True):
        super(ThisNewNet, self).__init__()
        self.in_plane_upsampler = UNet(in_channels=in_channels, n_classes=n_classes, depth=depth, wf=round(math.log(num_features,2)), batch_norm=batch_norm, up_mode=up_mode, dropout=dropout)
        self.slice_upsampler = SRCNN3D(n_channels=in_channels, scale_factor=scale_factor, num_features=num_features) 
        self.sliceup_first = sliceup_first
        self.loss_func = loss_func
        self.scale_factor = scale_factor
        self.loss_slice_count = loss_slice_count
        self.loss_inplane = loss_inplane

    def forward(self, images, gt=None):
        if self.sliceup_first:
            _, up_images = self.slice_upsampler(images)
            output = self.in_plane_upsampler(up_images)
        else:
            up_images = self.in_plane_upsampler(images)
            aux_out, output = self.slice_upsampler(up_images)

        if gt is None or self.loss_func is None:
            return output
        else:
            if self.sliceup_first:
                loss = self.loss_func(output, gt)
            else:
                in_plane_loss = self.loss_func(up_images, gt[:,:,::self.scale_factor[0],...]) #unet loss
                slice_aux_loss = self.loss_func(aux_out, gt) #aux srcnn loss
                slice_main_loss = self.loss_func(output, gt) #srcnn loss
                loss = slice_main_loss
                if self.loss_inplane:
                    loss += in_plane_loss
                if self.loss_slice_count > 1:
                    loss += slice_aux_loss
                # loss = in_plane_loss + slice_aux_loss + slice_main_loss        
            return output, loss