import torch
import torch.nn as nn
from .basic_model import BasicModel
from .Snet import Snet


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(BasicModel):
    def __init__(self, block, num_block, num_classes, expansion, snet):
        super().__init__()
        self.snet = snet
        self.in_channels = 64
        self.num_classes = num_classes
        self.expansion = expansion
        self.conv1 = self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, snet)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, snet)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, snet)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, snet)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, snet):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
            if self.snet:
                layers.append(Snet(out_channels * self.expansion**2 * block.expansion, self.num_classes))
        return nn.Sequential(*layers)

    def _normal_train_forward(self, x):
        assert self.snet == False
        y = []
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output, y

    def _bottleneck_mac(self, x, block):
        layer_num = 0
        tmp_mac = 0
        residual_func = block.residual_function
        short_cut = block.shortcut
        res = short_cut(x)
        if len(short_cut) != 0:
            self.statistic.conv_flops(res, 1, block.in_channels)
            tmp_mac += self.statistic.conv_mac(res, 1, block.in_channels)
        for layer in residual_func:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if layer_num == 0:
                    self.statistic.conv_flops(x, 1, block.in_channels)
                    tmp_mac += self.statistic.conv_mac(x, 1, block.in_channels)
                elif layer_num == 1:
                    self.statistic.conv_flops(x, 3, block.out_channels)
                    tmp_mac += self.statistic.conv_mac(x, 3, block.out_channels)
                else:
                    self.statistic.conv_flops(x, 1, block.out_channels)
                    tmp_mac += self.statistic.conv_mac(x, 1, block.out_channels)
                layer_num += 1
            elif isinstance(layer, nn.ReLU):
                self.statistic.relu_flops(x)

        x = x + res
        self.statistic.relu_flops(x)
        x = nn.functional.relu(x, True)
        self.statistic.relu_flops(x)
        return x, tmp_mac

    def _snet_mac(self, x, block):
        tmp_pool = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
        self.statistic.pooling_flops(x, self.expansion)
        self.statistic.linear_flops(torch.numel(x), self.num_classes)
        self.statistic.linear_mac(torch.numel(x), self.num_classes)
        tmp = block(tmp_pool)
        self.exit_strategy(tmp)

    def _normal_test_mac_forward(self, x):
        assert self.statistic is not None
        y = []
        for layer in self.conv1:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.statistic.conv_flops(x, 3, 3)
                self.statistic.conv_mac(x, 3, 3)
            elif isinstance(layer, nn.ReLU):
                self.statistic.relu_flops(x)

        for block in self.conv2_x:
            x, _ = self._bottleneck_mac(x, block)

        for block in self.conv3_x:
            x, _ = self._bottleneck_mac(x, block)

        for block in self.conv4_x:
            x, _ = self._bottleneck_mac(x, block)

        for block in self.conv5_x:
            x, _ = self._bottleneck_mac(x, block)

        x = self.avg_pool(x)
        self.statistic.pooling_flops(x, 1)
        x = x.view(x.size(0), -1)
        self.statistic.linear_flops(torch.numel(x), self.num_classes)
        x = self.fc(x)
        return x, y

    def _snet_test_mac_forward(self, x):
        assert self.snet == True
        assert self.statistic is not None
        assert self.exit_strategy is not None
        exit_point = 0
        y = []
        for layer in self.conv1:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.statistic.conv_flops(x, 3, 3)
                self.statistic.conv_mac(x, 3, 3)
            elif isinstance(layer, nn.ReLU):
                self.statistic.relu_flops(x)

        for block in self.conv2_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                if torch.sum(self.exit_strategy.res) == 1:
                    res = self.exit_strategy.res
                    self.exit_strategy.clear()
                    self.statistic.add_exit(layer=exit_point, category=torch.argmax(res.float()).item())
                    return res.float(), y
                else:
                    self.statistic.add_exclude(layer=exit_point, excluded=self.exit_strategy.res)
                exit_point += 1
            else:
                x, _ = self._bottleneck_mac(x, block)

        for block in self.conv3_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                if torch.sum(self.exit_strategy.res) == 1:
                    res = self.exit_strategy.res
                    self.exit_strategy.clear()
                    self.statistic.add_exit(layer=exit_point, category=torch.argmax(res.float()).item())
                    return res.float(), y
                else:
                    self.statistic.add_exclude(layer=exit_point, excluded=self.exit_strategy.res)
                exit_point += 1
            else:
                x, _ = self._bottleneck_mac(x, block)

        for block in self.conv4_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                if torch.sum(self.exit_strategy.res) == 1:
                    res = self.exit_strategy.res
                    self.exit_strategy.clear()
                    self.statistic.add_exit(layer=exit_point, category=torch.argmax(res.float()).item())
                    return res.float(), y
                else:
                    self.statistic.add_exclude(layer=exit_point, excluded=self.exit_strategy.res)
                exit_point += 1
            else:
                x, _ = self._bottleneck_mac(x, block)

        for block in self.conv5_x:
            if isinstance(block, Snet):
                if block == self.conv5_x[-1]:
                    continue
                self._snet_mac(x, block)
                if torch.sum(self.exit_strategy.res) == 1:
                    res = self.exit_strategy.res
                    self.exit_strategy.clear()
                    self.statistic.add_exit(layer=exit_point, category=torch.argmax(res.float()).item())
                    return res.float(), y
                else:
                    self.statistic.add_exclude(layer=exit_point, excluded=self.exit_strategy.res)
                exit_point += 1
            else:
                x, _ = self._bottleneck_mac(x, block)

        x = self.avg_pool(x)
        self.statistic.pooling_flops(x, 1)
        x = x.view(x.size(0), -1)
        self.statistic.linear_flops(torch.numel(x), self.num_classes)
        x = self.fc(x)
        return x, y

    def _snet_train_forward(self, x):
        assert self.snet == True
        y = []
        x = self.conv1(x)
        for block in self.conv2_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                y.append(tmp)
            else:
                x = block(x)

        for block in self.conv3_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                y.append(tmp)
            else:
                x = block(x)

        for block in self.conv4_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                y.append(tmp)
            else:
                x = block(x)

        for block in self.conv5_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                y.append(tmp)
            else:
                x = block(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, y[:-1]

    def _snet_test_fine_tune_forward(self, x):
        assert self.snet == True
        assert self.statistic is not None
        assert self.exit_strategy is not None
        exit_point = 0
        tmp_mac = 0
        y = []
        for layer in self.conv1:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                self.statistic.conv_flops(x, 3, 3)
                self.statistic.conv_mac(x, 3, 3)
            elif isinstance(layer, nn.ReLU):
                self.statistic.relu_flops(x)

        for block in self.conv2_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                self.statistic.add_mac_layer(exit_point, tmp_mac)
                exit_point += 1
            else:
                x, tmp_mac = self._bottleneck_mac(x, block)

        for block in self.conv3_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                self.statistic.add_mac_layer(exit_point, tmp_mac)
                exit_point += 1
            else:
                x, tmp_mac = self._bottleneck_mac(x, block)

        for block in self.conv4_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                self.statistic.add_mac_layer(exit_point, tmp_mac)
                exit_point += 1
            else:
                x, tmp_mac = self._bottleneck_mac(x, block)

        for block in self.conv5_x:
            if isinstance(block, Snet):
                self._snet_mac(x, block)
                self.statistic.add_mac_layer(exit_point, tmp_mac)
                exit_point += 1
            else:
                x, tmp_mac = self._bottleneck_mac(x, block)

        x = self.avg_pool(x)
        self.statistic.pooling_flops(x, 1)
        x = x.view(x.size(0), -1)
        self.statistic.linear_flops(torch.numel(x), self.num_classes)
        x = self.fc(x)
        return x, y

    def _snet_test_print_forward(self, x):
        assert self.snet == True
        y = []
        x = self.conv1(x)
        for block in self.conv2_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                print(torch.sigmoid(tmp))
                y.append(tmp)
            else:
                x = block(x)

        for block in self.conv3_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                print(torch.sigmoid(tmp))
                y.append(tmp)
            else:
                x = block(x)

        for block in self.conv4_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                print(torch.sigmoid(tmp))
                y.append(tmp)
            else:
                x = block(x)

        for block in self.conv5_x:
            if isinstance(block, Snet):
                tmp = torch.nn.functional.adaptive_avg_pool2d(x, self.expansion).flatten(1)
                tmp = block(tmp)
                print(torch.sigmoid(tmp))
                y.append(tmp)
            else:
                x = block(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, y[:-1]






