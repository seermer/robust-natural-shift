from torch import nn

import utils


# noinspection PyTypeChecker
class BasicResidual(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=3,
                 stride=1,
                 activation=nn.ReLU(inplace=True),
                 preact=False,
                 is_stem=False,
                 **kwargs):
        super(BasicResidual, self).__init__()
        residual = []
        if preact:
            if is_stem:
                self.pre_residual = nn.Sequential(
                    nn.BatchNorm2d(in_channels),
                    activation
                )
            else:
                residual.append(nn.BatchNorm2d(in_channels))
                residual.append(activation)
                self.pre_residual = None
        else:
            self.pre_residual = None
        residual.append(nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=utils.pad_ignore_stride(kernel_size),
                                  bias=False))
        residual.append(nn.BatchNorm2d(out_channels))
        residual.append(activation)
        residual.append(nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=utils.pad_ignore_stride(kernel_size),
                                  bias=preact))
        if not preact:
            residual.append(nn.BatchNorm2d(out_channels))
        self.residual = nn.Sequential(*residual)
        self.activation = None if preact else activation
        if (stride == 1 or stride == (1, 1)) and in_channels == out_channels:
            self.shortcut = None
        else:
            shortcut = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=preact)]
            if not preact:
                shortcut.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*shortcut)

    def forward(self, inputs):
        if self.pre_residual is not None:
            inputs = self.pre_residual(inputs)
        if self.shortcut is None:
            shortcut = inputs
        else:
            shortcut = self.shortcut(inputs)
        residual = self.residual(inputs)
        added = shortcut + residual
        if self.activation is None:
            return added
        else:
            return self.activation(added)


# noinspection PyTypeChecker
class Bottleneck(BasicResidual):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=3,
                 stride=1,
                 activation=nn.ReLU(inplace=True),
                 squeeze_factor=4,
                 preact=False,
                 is_stem=False):
        super(Bottleneck, self).__init__(in_channels, out_channels, kernel_size, stride, activation, preact, is_stem)
        residual = []
        if preact and not is_stem:
            residual.append(nn.BatchNorm2d(in_channels))
            residual.append(activation)
        squeezed = out_channels // squeeze_factor
        residual.append(nn.Conv2d(in_channels=in_channels,
                                  out_channels=squeezed,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False))
        residual.append(nn.BatchNorm2d(squeezed))
        residual.append(activation)
        residual.append(nn.Conv2d(in_channels=squeezed,
                                  out_channels=squeezed,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=utils.pad_ignore_stride(kernel_size),
                                  bias=False))
        residual.append(nn.BatchNorm2d(squeezed))
        residual.append(activation)
        residual.append(nn.Conv2d(in_channels=squeezed,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=preact))
        if not preact:
            residual.append(nn.BatchNorm2d(out_channels))
        self.residual = nn.Sequential(*residual)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        return self.pool(inputs).view(inputs.size()[0], -1)
