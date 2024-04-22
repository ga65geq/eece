import abc
import torch
import torch.nn as nn
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.exit_strategy = None
        self.exit_activation = None
        self.statistic = None
        self._forward = None

    def set_exit_strategy(self, strategy):
        self.exit_strategy = strategy

    def set_exit_activation(self, activation):
        self.exit_activation = activation

    def set_statistic(self, statistic):
        self.statistic = statistic

    def set_forward(self, forward_type):
        if forward_type == "normal_train":
            self._forward = self._normal_train_forward
        elif forward_type == "normal_test_mac":
            self._forward = self._normal_test_mac_forward
        elif forward_type == "normal_test_time":
            self._forward = self._normal_test_time_forward
        elif forward_type == "snet_train":
            self._forward = self._snet_train_forward
        elif forward_type == "snet_test_print":
            self.forward = self._snet_test_print_forward
        elif forward_type == "snet_test_time":
            self.forward = self._snet_test_time_forward
        elif forward_type == "snet_test_mac":
            self.forward = self._snet_test_mac_forward
        elif forward_type == "snet_test_fine_tune":
            self.forward = self._snet_test_fine_tune_forward
        else:
            raise NotImplementedError

    def _normal_train_forward(self, x):
        pass

    def _normal_test_mac_forward(self, x):
        pass

    def _normal_test_time_forward(self, x):
        pass

    def _snet_train_forward(self, x):
        pass

    def _snet_test_print_forward(self, x):
        pass

    def _snet_test_time_forward(self, x):
        pass

    def _snet_test_mac_forward(self, x):
        pass

    def _snet_test_fine_tune_forward(self, x):
        pass

    def forward(self, x):
        x, y = self._forward(x)
        return x, y

