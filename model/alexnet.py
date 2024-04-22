import torch
import torch.nn as nn

from .basic_model import BasicModel
from .Snet import Snet

num_channel = [64, 192, 384, 256, 256]


class AlexNet(BasicModel):
    def __init__(
        self, num_classes: int, dropout: float, snet: bool, expansion: int = 1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_snet = snet
        self.expansion = expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        if snet:
            self.snet = nn.Sequential(
                *[
                    Snet(in_features * expansion**2, num_classes)
                    for in_features in num_channel
                ]
            )

    def _normal_train_forward(self, x):
        y = []
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x, y

    def _normal_test_mac_forward(self, x):
        # Batch size is 1
        assert self.statistic is not None
        y = []
        x = self.conv1(x)
        self.statistic.conv_mac(x, 11, 3)
        self.statistic.conv_flops(x, 11, 3)
        x = self.relu1(x)
        self.statistic.relu_flops(x)
        x = self.pool1(x)
        self.statistic.pooling_flops(x, 3)

        x = self.conv2(x)
        self.statistic.conv_mac(x, 5, 64)
        self.statistic.conv_flops(x, 5, 64)
        x = self.relu2(x)
        self.statistic.relu_flops(x)
        x = self.pool2(x)
        self.statistic.pooling_flops(x, 3)

        x = self.conv3(x)
        self.statistic.conv_mac(x, 3, 192)
        self.statistic.conv_flops(x, 3, 192)
        x = self.relu3(x)
        self.statistic.relu_flops(x)

        x = self.conv4(x)
        self.statistic.conv_mac(x, 3, 384)
        self.statistic.conv_flops(x, 3, 384)
        x = self.relu4(x)
        self.statistic.relu_flops(x)

        x = self.conv5(x)
        self.statistic.conv_mac(x, 3, 256)
        self.statistic.conv_flops(x, 3, 256)
        x = self.relu5(x)
        self.statistic.relu_flops(x)
        x = self.pool5(x)
        self.statistic.pooling_flops(x, 3)

        x = self.avgpool(x)
        self.statistic.pooling_flops(x, 6)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        self.statistic.linear_mac(9216, 4096)
        self.statistic.linear_flops(9216, 4096)
        self.statistic.flops += 4096
        self.statistic.linear_mac(4096, 1024)
        self.statistic.linear_flops(4096, 1024)
        self.statistic.flops += 1024
        self.statistic.linear_mac(1024, self.num_classes)
        self.statistic.linear_flops(1024, self.num_classes)
        return x, y

    def _snet_train_forward(self, x):
        assert self.snet is not None
        y = []
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        y.append(
            self.snet[0](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        y.append(
            self.snet[1](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )

        x = self.conv3(x)
        x = self.relu3(x)
        y.append(
            self.snet[2](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )

        x = self.conv4(x)
        x = self.relu4(x)
        y.append(
            self.snet[3](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        y.append(
            self.snet[4](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x, y

    def _snet_test_print_forward(self, x):
        # batch size should be 1
        assert self.exit_activation is not None
        assert self.snet is not None
        y = []
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        y.append(
            self.snet[0](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )
        print("\n")
        print(self.exit_activation(y[-1]))

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        y.append(
            self.snet[1](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )
        print(self.exit_activation(y[-1]))

        x = self.conv3(x)
        x = self.relu3(x)
        y.append(
            self.snet[2](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )
        print(self.exit_activation(y[-1]))

        x = self.conv4(x)
        x = self.relu4(x)
        y.append(
            self.snet[3](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )
        print(self.exit_activation(y[-1]))

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        y.append(
            self.snet[4](
                nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
                    start_dim=1
                )
            )
        )
        print(self.exit_activation(y[-1]))

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x, y

    def _snet_test_mac_forward(self, x):
        assert self.snet is not None
        assert self.exit_strategy is not None
        assert self.statistic is not None
        y = []
        x = self.conv1(x)
        self.statistic.conv_mac(x, 11, 3)
        self.statistic.conv_flops(x, 11, 3)
        x = self.relu1(x)
        self.statistic.relu_flops(x)
        x = self.pool1(x)
        self.statistic.pooling_flops(x, 3)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[0](tmp_pool)
        y.append(tmp)
        self.exit_strategy(tmp)
        if torch.sum(self.exit_strategy.res) == 1:
            res = self.exit_strategy.res
            self.exit_strategy.clear()
            self.statistic.add_exit(layer=0, category=torch.argmax(res.float()).item())
            return res.float(), y
        else:
            self.statistic.add_exclude(layer=0, excluded=self.exit_strategy.res)

        x = self.conv2(x)
        self.statistic.conv_mac(x, 5, 64)
        self.statistic.conv_flops(x, 5, 64)
        x = self.relu2(x)
        self.statistic.relu_flops(x)
        x = self.pool2(x)
        self.statistic.pooling_flops(x, 3)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[1](tmp_pool)
        y.append(tmp)
        self.exit_strategy(tmp)
        if torch.sum(self.exit_strategy.res) == 1:
            res = self.exit_strategy.res
            self.exit_strategy.clear()
            self.statistic.add_exit(layer=1, category=torch.argmax(res.float()).item())
            return res.float(), y
        else:
            self.statistic.add_exclude(layer=1, excluded=self.exit_strategy.res)

        x = self.conv3(x)
        self.statistic.conv_mac(x, 3, 192)
        self.statistic.conv_flops(x, 3, 192)
        x = self.relu3(x)
        self.statistic.relu_flops(x)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[2](tmp_pool)
        y.append(tmp)
        self.exit_strategy(tmp)
        if torch.sum(self.exit_strategy.res) == 1:
            res = self.exit_strategy.res
            self.exit_strategy.clear()
            self.statistic.add_exit(layer=2, category=torch.argmax(res.float()).item())
            return res.float(), y
        else:
            self.statistic.add_exclude(layer=2, excluded=self.exit_strategy.res)

        x = self.conv4(x)
        self.statistic.conv_mac(x, 3, 384)
        self.statistic.conv_flops(x, 3, 384)
        x = self.relu4(x)
        self.statistic.relu_flops(x)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[3](tmp_pool)
        y.append(tmp)
        self.exit_strategy(tmp)
        if torch.sum(self.exit_strategy.res) == 1:
            res = self.exit_strategy.res
            self.exit_strategy.clear()
            self.statistic.add_exit(layer=3, category=torch.argmax(res.float()).item())
            return res.float(), y
        else:
            self.statistic.add_exclude(layer=3, excluded=self.exit_strategy.res)

        x = self.conv5(x)
        self.statistic.conv_mac(x, 3, 256)
        self.statistic.conv_flops(x, 3, 256)
        x = self.relu5(x)
        self.statistic.relu_flops(x)
        x = self.pool5(x)
        self.statistic.pooling_flops(x, 3)

        x = self.avgpool(x)
        self.statistic.pooling_flops(x, 6)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[4](tmp_pool)
        y.append(tmp)
        self.exit_strategy(tmp)
        if torch.sum(self.exit_strategy.res) == 1:
            res = self.exit_strategy.res
            self.exit_strategy.clear()
            self.statistic.add_exit(layer=4, category=torch.argmax(res.float()).item())
            return res.float(), y
        else:
            self.statistic.add_exclude(layer=4, excluded=self.exit_strategy.res)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        self.statistic.linear_mac(9216, 4096)
        self.statistic.linear_flops(9216, 4096)
        self.statistic.flops += 4096
        self.statistic.linear_mac(4096, 1024)
        self.statistic.linear_flops(4096, 1024)
        self.statistic.flops += 1024
        self.statistic.linear_mac(1024, self.num_classes)
        self.statistic.linear_flops(1024, self.num_classes)
        return x, y

    def _snet_test_fine_tune_forward(self, x):
        assert self.snet is not None
        assert self.exit_strategy is not None
        assert self.statistic is not None
        y = []
        x = self.conv1(x)
        tmp_mac = self.statistic.conv_mac(x, 11, 3)
        self.statistic.conv_flops(x, 11, 3)
        x = self.relu1(x)
        self.statistic.relu_flops(x)
        x = self.pool1(x)
        self.statistic.pooling_flops(x, 3)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[0](tmp_pool)
        y.append(tmp)
        self.statistic.add_mac_layer(0, tmp_mac)
        self.exit_strategy(tmp)

        x = self.conv2(x)
        tmp_mac = self.statistic.conv_mac(x, 5, 64)
        self.statistic.conv_flops(x, 5, 64)
        x = self.relu2(x)
        self.statistic.relu_flops(x)
        x = self.pool2(x)
        self.statistic.pooling_flops(x, 3)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[1](tmp_pool)
        y.append(tmp)
        self.statistic.add_mac_layer(1, tmp_mac)
        self.exit_strategy(tmp)

        x = self.conv3(x)
        self.statistic.conv_mac(x, 3, 192)
        tmp_mac = self.statistic.conv_flops(x, 3, 192)
        x = self.relu3(x)
        self.statistic.relu_flops(x)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[2](tmp_pool)
        y.append(tmp)
        self.statistic.add_mac_layer(2, tmp_mac)
        self.exit_strategy(tmp)

        x = self.conv4(x)
        tmp_mac = self.statistic.conv_mac(x, 3, 384)
        self.statistic.conv_flops(x, 3, 384)
        x = self.relu4(x)
        self.statistic.relu_flops(x)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[3](tmp_pool)
        y.append(tmp)
        self.statistic.add_mac_layer(3, tmp_mac)
        self.exit_strategy(tmp)

        x = self.conv5(x)
        tmp_mac = self.statistic.conv_mac(x, 3, 256)
        self.statistic.conv_flops(x, 3, 256)
        x = self.relu5(x)
        self.statistic.relu_flops(x)
        x = self.pool5(x)
        self.statistic.pooling_flops(x, 3)

        x = self.avgpool(x)
        self.statistic.pooling_flops(x, 6)

        tmp_pool = nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(
            start_dim=1
        )
        self.statistic.pooling_flops(tmp_pool, self.expansion)
        self.statistic.linear_mac(torch.numel(tmp_pool), self.num_classes)
        self.statistic.linear_flops(torch.numel(tmp_pool), self.num_classes)
        tmp = self.snet[4](tmp_pool)
        y.append(tmp)
        self.statistic.add_mac_layer(4, tmp_mac)
        self.exit_strategy(tmp)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        tmp_mac = self.statistic.linear_mac(9216, 4096)
        self.statistic.linear_flops(9216, 4096)
        tmp_mac += self.statistic.linear_mac(4096, 1024)
        self.statistic.linear_flops(4096, 1024)
        tmp_mac += self.statistic.linear_mac(1024, self.num_classes)
        self.statistic.linear_flops(1024, self.num_classes)
        self.statistic.add_mac_layer(5, tmp_mac)
        return x, y

    def load_pretrained_model(self, path: str):
        state_dict = torch.load(path)
        self.conv1.weight = state_dict["features.0.weight"]
        self.conv1.bias = state_dict["features.0.bias"]
        self.conv2.weight = state_dict["features.3.weight"]
        self.conv2.bias = state_dict["features.3.bias"]
        self.conv3.weight = state_dict["features.6.weight"]
        self.conv3.bias = state_dict["features.6.bias"]
        self.conv4.weight = state_dict["features.8.weight"]
        self.conv4.bias = state_dict["features.8.bias"]
        self.conv5.weight = state_dict["features.10.weight"]
        self.conv5.bias = state_dict["features.10.bias"]
