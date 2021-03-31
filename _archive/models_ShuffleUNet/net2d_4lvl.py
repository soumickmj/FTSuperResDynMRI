import torch
import torch.nn as nn

from . import pixel_shuffle, pixel_unshuffle

# -------------------------------------------------------------------------------------------------------------------------------------------------##

class _double_conv(nn.Module):
    """
    Double Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(_double_conv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu((x))
        x = self.conv_2(x)
        x = self.relu((x))

        return x


class _conv_decomp(nn.Module):
    """
    Convolutional Decomposition Block
    """

    def __init__(self, in_channels, out_channels, k_size, stride, bias=True):
        super(_conv_decomp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                      stride=stride, padding=k_size // 2, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu((x1))
        x2 = self.conv2(x)
        x2 = self.relu((x2))
        x3 = self.conv3(x)
        x3 = self.relu((x3))
        x4 = self.conv4(x)
        x4 = self.relu((x4))
        return x1, x2, x3, x4


class _concat(nn.Module):
    """
    Skip-Addition block
    """

    def __init__(self):
        super(_concat, self).__init__()

    def forward(self, e1, e2, e3, e4, d1, d2, d3, d4):
        self.X1 = e1 + d1
        self.X2 = e2 + d2
        self.X3 = e3 + d3
        self.X4 = e4 + d4
        x = torch.cat([self.X1, self.X2, self.X3, self.X4], dim=1)

        return x

# -------------------------------------------------------------------------------------------------------------------------------------------------##

class ShuffleUNet(nn.Module):

    def __init__(self, in_ch=1, num_features=64, out_ch=1, kernel_size_1=3, stride_1=1, kernel_size_3=1):
        super(ShuffleUNet, self).__init__()

        num_features = num_features
        filters = [num_features, num_features * 2, num_features * 4, num_features * 8, num_features * 16]

        #Pixel Shuffles
        self.pixel_1 = pixel_shuffle.PixelShuffle(num_features * 16, num_features * 16 * 2, kernel_size_1, stride_1, d=2)
        self.pixel_2 = pixel_shuffle.PixelShuffle(num_features * 8, num_features * 8 * 2, kernel_size_1, stride_1, d=2)
        self.pixel_3 = pixel_shuffle.PixelShuffle(num_features * 4, num_features * 4 * 2, kernel_size_1, stride_1, d=2)
        self.pixel_4 = pixel_shuffle.PixelShuffle(num_features * 2, num_features * 2 * 2, kernel_size_1, stride_1, d=2)

        #Pixel Unshuffles
        self.pixel_unshuffle_1 = pixel_unshuffle.PixelUnshuffle(num_features, num_features, 3, 1, d=2)
        self.pixel_unshuffle_2 = pixel_unshuffle.PixelUnshuffle(num_features * 2, num_features * 2, 3, 1, d=2)
        self.pixel_unshuffle_3 = pixel_unshuffle.PixelUnshuffle(num_features * 4, num_features * 4, 3, 1, d=2)
        self.pixel_unshuffle_4 = pixel_unshuffle.PixelUnshuffle(num_features * 8, num_features * 8, 3, 1, d=2)

        #Contraction path
        self.conv_inp = _double_conv(in_ch, filters[0], kernel_size_1, stride_1)
        self.wave_1_down = _conv_decomp(filters[0], filters[0], kernel_size_1, stride_1)
        self.conv_enc_1 = _double_conv(filters[0], filters[1], kernel_size_1, stride_1)
        self.wave_2_down = _conv_decomp(filters[1], filters[1], kernel_size_1, stride_1)
        self.conv_enc_2 = _double_conv(filters[1], filters[2], kernel_size_1, stride_1)
        self.wave_3_down = _conv_decomp(filters[2], filters[2], kernel_size_1, stride_1)
        self.conv_enc_3 = _double_conv(filters[2], filters[3], kernel_size_1, stride_1)
        self.wave_4_down = _conv_decomp(filters[3], filters[3], kernel_size_1, stride_1)
        self.conv_enc_4 = _double_conv(filters[3], filters[4], kernel_size_1, stride_1)

        #Expansion path
        self.wave_4_up = _conv_decomp(filters[3], filters[3], kernel_size_1, stride_1)
        self.wave_3_up = _conv_decomp(filters[2], filters[2], kernel_size_1, stride_1)
        self.wave_2_up = _conv_decomp(filters[1], filters[1], kernel_size_1, stride_1)
        self.wave_1_up = _conv_decomp(filters[0], filters[0], kernel_size_1, stride_1)
        self.cat = _concat()
        self.convup_4 = _double_conv(filters[3] * 5, filters[3], kernel_size_1, stride_1)
        self.convup_3 = _double_conv(filters[2] * 5, filters[2], kernel_size_1, stride_1)
        self.convup_2 = _double_conv(filters[1] * 5, filters[1], kernel_size_1, stride_1)
        self.convup_1 = _double_conv(filters[0] * 5, filters[0], kernel_size_1, stride_1)

        #FC
        self.out = nn.Conv2d(filters[0], out_ch, kernel_size=kernel_size_3, stride=stride_1, padding=0, bias=True)

        #Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                weight = nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.weight.data.copy_(weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        e1 = self.conv_inp(x)
        wave_e1_1, wave_e1_2, wave_e1_3, wave_e1_4 = self.wave_1_down(e1)
        e2 = self.pixel_unshuffle_1(wave_e1_4)

        e3 = self.conv_enc_1(e2)
        wave_e2_1, wave_e2_2, wave_e2_3, wave_e2_4 = self.wave_2_down(e3)
        e4 = self.pixel_unshuffle_2(wave_e2_4)

        e5 = self.conv_enc_2(e4)
        wave_e3_1, wave_e3_2, wave_e3_3, wave_e3_4 = self.wave_3_down(e5)
        e6 = self.pixel_unshuffle_3(wave_e3_4)

        e7 = self.conv_enc_3(e6)
        wave_e4_1, wave_e4_2, wave_e4_3, wave_e4_4 = self.wave_4_down(e7)
        e8 = self.pixel_unshuffle_4(wave_e4_4)

        e9 = self.conv_enc_4(e8)

        d4 = self.pixel_1(e9)
        wave_d4_1, wave_d4_2, wave_d4_3, wave_d4_4 = self.wave_4_up(d4)
        cat_4 = self.cat(wave_e4_1, wave_e4_2, wave_e4_3, wave_e4_4, wave_d4_1, wave_d4_2, wave_d4_3, wave_d4_4)
        d4 = self.convup_4(torch.cat([e7, cat_4], dim=1))

        d3 = self.pixel_2(d4)
        wave_d3_1, wave_d3_2, wave_d3_3, wave_d3_4 = self.wave_3_up(d3)
        cat_3 = self.cat(wave_e3_1, wave_e3_2, wave_e3_3, wave_e3_4, wave_d3_1, wave_d3_2, wave_d3_3, wave_d3_4)
        d3 = self.convup_3(torch.cat([e5, cat_3], dim=1))

        d2 = self.pixel_3(d3)
        wave_d2_1, wave_d2_2, wave_d2_3, wave_d2_4 = self.wave_2_up(d2)
        cat_2 = self.cat(wave_e2_1, wave_e2_2, wave_e2_3, wave_e2_4, wave_d2_1, wave_d2_2, wave_d2_3, wave_d2_4)
        d2 = self.convup_2(torch.cat([e3, cat_2], dim=1))

        d1 = self.pixel_4(d2)
        wave_d1_1, wave_d1_2, wave_d1_3, wave_d1_4 = self.wave_1_up(d1)
        cat_1 = self.cat(wave_e1_1, wave_e1_2, wave_e1_3, wave_e1_4, wave_d1_1, wave_d1_2, wave_d1_3, wave_d1_4)
        d1 = self.convup_1(torch.cat([e1, cat_1], dim=1))

        out = self.out(d1)
        return out
