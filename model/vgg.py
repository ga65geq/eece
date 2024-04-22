from typing import cast

import torch
import torch.nn as nn

from .basic_model import BasicModel
from .Snet import Snet


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


cfg = {
    "M": [128, 128, "M", 256, 256, "M", 512, 512, "M"],
    "MS": [128, "S", 128, "S", "M", 256, "S", 256, "S", "M", 512, "S", 512, "S", "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "BS": [
        64,
        "S",
        64,
        "S",
        "M",
        128,
        "S",
        128,
        "S",
        "M",
        256,
        "S",
        256,
        "S",
        "M",
        512,
        "S",
        512,
        "S",
        "M",
        512,
        "S",
        512,
        "S",
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(BasicModel):
    def __init__(self, features: nn.Sequential, num_classes: int, expansion: int):
        super().__init__()
        self.features = features
        self.expansion = expansion
        self.num_classes = num_classes
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(True),
        #     nn.Linear(4096, num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )

    def _normal_train_forward(self, x):
        y = []
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, y

    def _normal_test_mac_forward(self, x):
        assert self.statistic is not None
        layer_num = 0
        y = []
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                inchannel = self.statistic.remained_channel(x)
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if layer_num == 0:
                    self.statistic.conv_flops(x, 3, inchannel)
                    self.statistic.conv_mac(x, 3, inchannel)
                elif layer_num == 1:
                    self.statistic.conv_mac(x, 3, inchannel)
                    self.statistic.conv_flops(x, 3, inchannel)
                elif layer_num < 4:
                    self.statistic.conv_mac(x, 3, inchannel)
                    self.statistic.conv_flops(x, 3, inchannel)
                else:
                    self.statistic.conv_mac(x, 3, inchannel)
                    self.statistic.conv_flops(x, 3, inchannel)
                layer_num += 1
            elif isinstance(layer, nn.ReLU):
                self.statistic.relu_flops(x)
            elif isinstance(layer, nn.MaxPool2d):
                self.statistic.pooling_flops(x, 2)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        self.statistic.linear_mac(8192, 1024)
        self.statistic.flops += 1024
        self.statistic.linear_flops(8192, 1024)
        self.statistic.linear_mac(1024, 1024)
        self.statistic.linear_flops(1024, 1024)
        self.statistic.flops += 1024
        self.statistic.linear_mac(1024, self.num_classes)
        self.statistic.linear_flops(1024, self.num_classes)
        return x, y

    # def _normal_test_mac_forward(self, x):
    #     assert self.statistic is not None
    #     layer_num = 0
    #     y = []
    #     for layer in self.features:
    #         x = layer(x)
    #         if isinstance(layer, nn.Conv2d):
    #             if layer_num == 0:
    #                 self.statistic.conv_flops(x, 3, 3)
    #                 self.statistic.conv_mac(x, 3, 3)
    #             elif layer_num == 1:
    #                 self.statistic.conv_mac(x, 3, 128)
    #                 self.statistic.conv_flops(x, 3, 128)
    #             elif layer_num < 4:
    #                 self.statistic.conv_mac(x, 3, 256)
    #                 self.statistic.conv_flops(x, 3, 256)
    #             else:
    #                 self.statistic.conv_mac(x, 3, 512)
    #                 self.statistic.conv_flops(x, 3, 512)
    #             layer_num += 1
    #         elif isinstance(layer, nn.ReLU):
    #             self.statistic.relu_flops(x)
    #         elif isinstance(layer, nn.MaxPool2d):
    #             self.statistic.pooling_flops(x, 2)
    #
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     self.statistic.linear_mac(8192, 1024)
    #     self.statistic.flops += 1024
    #     self.statistic.linear_flops(8192, 1024)
    #     self.statistic.linear_mac(1024, 1024)
    #     self.statistic.linear_flops(1024, 1024)
    #     self.statistic.flops += 1024
    #     self.statistic.linear_mac(1024, self.num_classes)
    #     self.statistic.linear_flops(1024, self.num_classes)
    #     return x, y

    def _snet_train_forward(self, x):
        y = []
        for layer in self.features:
            if isinstance(layer, Snet):
                tmp = layer(
                    torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                        1
                    )
                )
                y.append(tmp)
            else:
                x = layer(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, y

    def _snet_test_print_forward(self, x):
        assert self.exit_activation is not None
        y = []
        for layer in self.features:
            if isinstance(layer, Snet):
                tmp = layer(
                    torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                        1
                    )
                )
                y.append(tmp)
                print(self.exit_activation(y[-1]))
            else:
                x = layer(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, y

    def _snet_test_mac_forward(self, x):
        assert self.exit_strategy is not None
        assert self.statistic is not None
        layer_num = 0
        exit_point = 0
        self.exit_strategy.clear()
        y = []
        for layer in self.features:
            inchannel = 3
            if isinstance(layer, Snet):
                tmp_pool = torch.nn.functional.adaptive_avg_pool2d(
                    x, self.expansion
                ).flatten(1)
                self.statistic.pooling_flops(tmp_pool, self.expansion)
                self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
                self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
                tmp = layer(tmp_pool)
                self.exit_strategy(tmp)
                if torch.sum(self.exit_strategy.res) == 1:
                    res = self.exit_strategy.res
                    self.exit_strategy.clear()
                    self.statistic.add_exit(
                        layer=exit_point, category=torch.argmax(res.float()).item()
                    )
                    return res.float(), y
                else:
                    self.statistic.add_exclude(
                        layer=exit_point, excluded=self.exit_strategy.res
                    )
                exit_point += 1
            else:
                if isinstance(layer, nn.Conv2d):
                    inchannel = self.statistic.remained_channel(x)
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    if layer_num == 0:
                        self.statistic.conv_flops(x, 3, inchannel)
                        self.statistic.conv_mac(x, 3, inchannel)
                    elif layer_num == 1:
                        self.statistic.conv_mac(x, 3, inchannel)
                        self.statistic.conv_flops(x, 3, inchannel)
                    elif layer_num < 4:
                        self.statistic.conv_mac(x, 3, inchannel)
                        self.statistic.conv_flops(x, 3, inchannel)
                    else:
                        self.statistic.conv_mac(x, 3, inchannel)
                        self.statistic.conv_flops(x, 3, inchannel)
                    layer_num += 1
                elif isinstance(layer, nn.ReLU):
                    self.statistic.relu_flops(x)
                elif isinstance(layer, nn.MaxPool2d):
                    self.statistic.pooling_flops(x, 2)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        self.statistic.linear_mac(8192, 1024)
        self.statistic.flops += 1024
        self.statistic.linear_flops(8192, 1024)
        self.statistic.linear_mac(1024, 1024)
        self.statistic.linear_flops(1024, 1024)
        self.statistic.flops += 1024
        self.statistic.linear_mac(1024, self.num_classes)
        self.statistic.linear_flops(1024, self.num_classes)
        return x, y

    # def _snet_test_mac_forward(self, x):
    #     assert self.exit_strategy is not None
    #     assert self.statistic is not None
    #     layer_num = 0
    #     exit_point = 0
    #     self.exit_strategy.clear()
    #     y = []
    #     for layer in self.features:
    #         if isinstance(layer, Snet):
    #             tmp_pool = torch.nn.functional.adaptive_avg_pool2d(
    #                 x, self.expansion
    #             ).flatten(1)
    #             self.statistic.pooling_flops(tmp_pool, self.expansion)
    #             self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
    #             self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
    #             tmp = layer(tmp_pool)
    #             self.exit_strategy(tmp)
    #             if torch.sum(self.exit_strategy.res) == 1:
    #                 res = self.exit_strategy.res
    #                 self.exit_strategy.clear()
    #                 self.statistic.add_exit(
    #                     layer=exit_point, category=torch.argmax(res.float()).item()
    #                 )
    #                 return res.float(), y
    #             else:
    #                 self.statistic.add_exclude(
    #                     layer=exit_point, excluded=self.exit_strategy.res
    #                 )
    #             exit_point += 1
    #         else:
    #             x = layer(x)
    #             if isinstance(layer, nn.Conv2d):
    #                 if layer_num == 0:
    #                     self.statistic.conv_flops(x, 3, 3)
    #                     self.statistic.conv_mac(x, 3, 3)
    #                 elif layer_num == 1:
    #                     self.statistic.conv_mac(x, 3, 128)
    #                     self.statistic.conv_flops(x, 3, 128)
    #                 elif layer_num < 4:
    #                     self.statistic.conv_mac(x, 3, 256)
    #                     self.statistic.conv_flops(x, 3, 256)
    #                 else:
    #                     self.statistic.conv_mac(x, 3, 512)
    #                     self.statistic.conv_flops(x, 3, 512)
    #                 layer_num += 1
    #             elif isinstance(layer, nn.ReLU):
    #                 self.statistic.relu_flops(x)
    #             elif isinstance(layer, nn.MaxPool2d):
    #                 self.statistic.pooling_flops(x, 2)
    #
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     self.statistic.linear_mac(8192, 1024)
    #     self.statistic.flops += 1024
    #     self.statistic.linear_flops(8192, 1024)
    #     self.statistic.linear_mac(1024, 1024)
    #     self.statistic.linear_flops(1024, 1024)
    #     self.statistic.flops += 1024
    #     self.statistic.linear_mac(1024, self.num_classes)
    #     self.statistic.linear_flops(1024, self.num_classes)
    #     return x, y

    #
    def _snet_test_fine_tune_forward(self, x):
        assert self.exit_strategy is not None
        assert self.statistic is not None
        layer_num = 0
        exit_point = 0
        self.exit_strategy.clear()
        y = []
        for layer in self.features:
            if isinstance(layer, Snet):
                tmp_pool = torch.nn.functional.adaptive_avg_pool2d(
                    x, self.expansion
                ).flatten(1)
                self.statistic.pooling_flops(tmp_pool, self.expansion)
                self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
                self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
                tmp = layer(tmp_pool)
                self.exit_strategy(tmp)
                exit_point += 1
            else:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    if layer_num == 0:
                        tmp_mac = self.statistic.conv_mac(x, 3, 3)
                        self.statistic.add_mac_layer(exit_point, tmp_mac)
                    elif layer_num == 1:
                        tmp_mac = self.statistic.conv_mac(x, 3, 128)
                        self.statistic.add_mac_layer(exit_point, tmp_mac)
                    elif layer_num < 4:
                        tmp_mac = self.statistic.conv_mac(x, 3, 256)
                        self.statistic.add_mac_layer(exit_point, tmp_mac)
                    else:
                        tmp_mac = self.statistic.conv_mac(x, 3, 512)
                        self.statistic.add_mac_layer(exit_point, tmp_mac)
                    layer_num += 1

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        tmp_mac = self.statistic.linear_mac(8192, 1024)
        tmp_mac += self.statistic.linear_mac(1024, 1024)
        tmp_mac += self.statistic.linear_mac(1024, self.num_classes)
        self.statistic.add_mac_layer(exit_point, tmp_mac)

        return x, y


def make_layers(cfg, batch_norm=True, output_features=100, expansion=1):
    layer = []
    input_channel = 3
    for idx, item in enumerate(cfg):
        if item == "M":
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        if item == "S":
            layer += [Snet(cast(int, cfg[idx - 1]) * expansion**2, output_features)]
            continue

        layer += [conv3x3(input_channel, cast(int, item))]

        if batch_norm:
            layer += [nn.BatchNorm2d(item)]

        layer += [nn.ReLU(True)]
        input_channel = item

    return nn.Sequential(*layer)


def vgg_factory(vgg_type, num_classes, expansion):
    return VGG(
        make_layers(cfg[vgg_type], True, num_classes, expansion),
        num_classes=num_classes,
        expansion=expansion,
    )
