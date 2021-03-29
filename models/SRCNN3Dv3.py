import numpy as np
import torch
import torch.nn as nn

__author__ = "Soumick Chatterjee, Geetha Doddapaneni Gopinath"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Geetha Doddapaneni Gopinath"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under Testing"

class SRCNN3Dv3(nn.Module):
    def __init__(self,n_channels=1,scale_factor=2,num_features=32,kernel_size=3,stride=1):
        super(SRCNN3Dv3, self).__init__()
        if type(scale_factor) is int:
            self.scale_factor=(scale_factor,scale_factor,scale_factor)
        else:
            self.scale_factor=scale_factor

        self.n_channels=n_channels

        self.conv_1 = nn.Sequential(nn.Conv3d(n_channels, num_features, kernel_size, stride, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, stride, padding=kernel_size // 2), 
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))

        self.conv_6s = nn.ModuleList()
        self.conv_6s_post = nn.ModuleList()
        for i in range(len(self.scale_factor)):
            if self.scale_factor[i] > 1:
                out_features = np.prod(self.scale_factor[i:])
                self.conv_6s.append(nn.Sequential(nn.Conv3d(num_features, out_features, kernel_size, stride, padding=kernel_size // 2), 
                                        nn.BatchNorm3d(num_features=out_features), nn.ReLU(inplace=True)))
                self.conv_6s_post.append(nn.Sequential(nn.Conv3d(out_features // self.scale_factor[i], num_features, kernel_size, stride, padding=kernel_size // 2), 
                                        nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True)))

        self.conv_7 = nn.Sequential(nn.Conv3d(n_channels, num_features, kernel_size, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_8 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_9 = nn.Sequential(nn.Conv3d(num_features, num_features, kernel_size, padding=kernel_size // 2),
                                    nn.BatchNorm3d(num_features=num_features), nn.ReLU(inplace=True))
        self.conv_10 = nn.Sequential(nn.Conv3d(num_features, n_channels, kernel_size, padding=kernel_size // 2),
                                     nn.Sigmoid())

    def forward(self, image):
        output_1 = self.conv_1(image)
        output_2 = self.conv_2(output_1)
        output_3a = self.conv_3(output_2)
        output_3 = torch.add(output_1, output_3a) #torch.mul(output_1, 1) = output_1 #Note for Geetha
        output_4 = self.conv_4(output_3)
        output_5a = self.conv_5(output_4)
        output_5 = torch.add(output_1, output_5a)
        
        output_6 = output_5
        mod_ind = 0
        for i in range(len(self.scale_factor)):
            if self.scale_factor[i] > 1:
                output_6 = self.conv_6s[mod_ind](output_6)
                suffled_size = list(output_6.shape)
                suffled_size[1] //= self.scale_factor[i]
                suffled_size[2+i] *= self.scale_factor[i]
                output_6 = output_6.view(suffled_size)
                if i+1 < len(self.scale_factor):
                    output_6 = self.conv_6s_post[mod_ind](output_6)
                mod_ind += 1
        
        output_7 = self.conv_7(output_6)
        output_8 = self.conv_8(output_7)
        output_9a = self.conv_9(output_8)
        output_9 = torch.add(output_7, output_9a)
        output = self.conv_10(output_9)  # Final Loss
        return  output_6, output


if __name__ == "__main__":
    tensor = torch.rand((2, 1, 24, 16, 16)).cuda()
    model = SRCNN3Dv2(scale_factor=(2,4,3)).cuda()
    model(tensor)

#     model = SRCNN3D(1,num_features=64,scale_factor=(2,1,1)).cuda()
#     from torchsummary import summary
#     summary(model, input_size=(1, 32, 32, 32))
