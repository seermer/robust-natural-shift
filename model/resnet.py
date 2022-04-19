from torch import nn
from collections import OrderedDict

import utils
from  model import modules


# noinspection PyTypeChecker
def get_resnet_model(in_channels=3, width_factor=1., num_classes=10, wide_resnet=0,
                     configs: list = None, preact=False, bottleneck=False, reduced_stem=True, activation=nn.ReLU(True)):
    if wide_resnet:
        preact = True
    if configs is None:
        if bottleneck:
            configs = [3, 4, 6, 3]
        else:
            if reduced_stem:
                # default to wrn28
                configs = [4, 4, 4]
            else:
                configs = [3, 4, 6, 3]
    model = []
    stem = []
    stem_filters = 3 if reduced_stem else 7
    curr_in_channels = int(64 * width_factor)

    squeeze_factor = 2 if wide_resnet else 4
    stem.append(nn.Conv2d(in_channels=in_channels,
                          out_channels=curr_in_channels,
                          kernel_size=stem_filters,
                          stride=1 if reduced_stem else 2,
                          bias=preact,
                          padding=utils.pad_ignore_stride(stem_filters)))
    if not preact:
        stem.append(nn.BatchNorm2d(curr_in_channels))
        stem.append(activation)
    if not reduced_stem:
        stem.append(nn.MaxPool2d(3, 2, padding=utils.pad_ignore_stride(3)))
    model.append(('Stem', nn.Sequential(*stem)))
    del stem, stem_filters, reduced_stem, in_channels, width_factor
    block = modules.Bottleneck if bottleneck else modules.BasicResidual
    curr_out_channels = curr_in_channels * (squeeze_factor if bottleneck else 1)
    if preact:
        if wide_resnet:
            curr_out_channels *= wide_resnet
        model.append(('ResBlock0_0',
                      block(curr_in_channels, curr_out_channels, activation=activation,
                            preact=True, is_stem=True, squeeze_factor=2 if wide_resnet else 4)))
        curr_in_channels = curr_out_channels
    for i in range(len(configs)):
        for j in range(1 if (preact and i == 0) else 0, configs[i]):
            if j == 0 and i != 0:
                curr_out_channels = curr_in_channels * 2
                model.append(('ResBlock{}_{}'.format(i, j),
                              block(curr_in_channels, curr_out_channels,
                                    stride=2, preact=preact, activation=activation,
                                    squeeze_factor=squeeze_factor)))

            else:
                if j > 0:
                    curr_in_channels = curr_out_channels
                elif wide_resnet and i == 0:
                    curr_out_channels *= wide_resnet
                model.append(('ResBlock{}_{}'.format(i, j),
                              block(curr_in_channels, curr_out_channels,
                                    preact=preact, activation=activation,
                                    squeeze_factor=squeeze_factor)))

    head = []
    if preact:
        head.append(nn.BatchNorm2d(curr_in_channels))
        head.append(activation)
    head.append(modules.GlobalAvgPool())
    model.append(('head', nn.Sequential(*head)))
    model.append(('Output', nn.Linear(curr_in_channels, num_classes)))
    model = OrderedDict(model)
    return nn.Sequential(*(model.values()))


if __name__ == '__main__':
    from torchinfo import summary

    summary(get_resnet_model(wide_resnet=10, reduced_stem=True, width_factor=.25), input_size=(256, 3, 32, 32),
            col_names=["output_size", 'num_params'],
            depth=1)
