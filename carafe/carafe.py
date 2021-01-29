import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import carafe_ext


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class CARAFEPack(nn.Module):
    """ A unified package of CARAFE upsampler that contains:
    1) channel compressor 2) content encoder 3) CARAFE op

    Official implementation of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 channels,
                 scale_factor,
                 up_kernel=5,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64):
        super(CARAFEPack, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels,
                                            1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / (self.up_kernel * self.up_kernel))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x: Tensor, mask: Tensor):
        x = self.forward_carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward_carafe(self, features: Tensor, masks: Tensor):
        assert self.scale_factor >= 1
        assert masks.size(1) == self.up_kernel * self.up_kernel * self.up_group
        assert masks.size(-1) == features.size(-1) * self.scale_factor
        assert masks.size(-2) == features.size(-2) * self.scale_factor
        assert features.size(1) % self.up_group == 0
        assert (self.up_kernel - 1) % 2 == 0 and self.up_kernel >= 1

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * self.scale_factor, w * self.scale_factor))
        routput = features.new_zeros(output.size())
        rfeatures = features.new_zeros(features.size())
        rmasks = masks.new_zeros(masks.size())
        if features.is_cuda:
            carafe_ext.forward(features, rfeatures, masks, rmasks, self.up_kernel,
                            self.up_group, self.scale_factor, routput, output)
        else:
            raise NotImplementedError

        return output

    def forward(self, x: Tensor):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        return x
