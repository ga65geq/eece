import torch


class Statistic:
    def __init__(self, num_layers, num_classes):
        # The total number of mul operation
        self.num_layers = num_layers
        self.flops = 0
        self.mac = 0
        # The num of mul operation in the next layer
        self.mac_layer = {i: 0 for i in range(num_layers + 1)}
        # The early exit statistic
        self.exit = {}
        for i in range(num_layers):
            self.exit[i] = {j: 0 for j in range(num_classes)}
        # The number of excluded category
        self.exclude = {}
        for i in range(num_layers):
            self.exclude[i] = {j: 0 for j in range(num_classes)}

    @staticmethod
    def remained_channel(x):
        tmp = torch.flatten(x, start_dim=2)
        tmp = torch.sum(tmp, 2)
        non_zero = len(torch.nonzero(tmp))
        return non_zero

    def conv_mac(self, x, kernel, in_channel):
        out_channel = self.remained_channel(x)
        num_element = out_channel * x.size()[0] * x.size()[2] * x.size()[3]
        tmp = num_element * kernel ** 2 * in_channel
        self.mac += tmp
        return tmp

    def linear_mac(self, cin, cout, n=1):
        tmp = cin * cout * n
        self.flops += tmp
        return tmp

    def conv_flops(self, x, kernel, in_channel):
        out_channel = self.remained_channel(x)
        num_element = out_channel * x.size()[0] * x.size()[2] * x.size()[3]
        tmp = num_element * kernel ** 2 * in_channel * 2
        self.flops += tmp
        return tmp

    def relu_flops(self, x):
        tmp = torch.numel(x)
        self.flops += tmp
        return tmp

    def pooling_flops(self, x, kernel):
        tmp = torch.numel(x) * kernel ** 2
        self.flops += tmp
        return tmp

    def linear_flops(self, cin, cout):
        tmp = 2 * cin * cout
        self.flops += tmp
        return tmp

    def add_exit(self, layer, category):
        self.exit[layer][category] += 1

    def add_mac_layer(self, layer, num):
        if self.mac_layer[layer] == 0:
            self.mac_layer[layer] = num
        else:
            return

    def add_exclude(self, layer, excluded):
        excluded = torch.where(excluded == 0)[1].tolist()
        for category in excluded:
            self.exclude[layer][category] += 1

    def print(self):
        print("Exit info:")
        self.get_exit_result()
        print("Exclude info:")
        self.get_exclude_result()
        print("Total mac:{}".format(self.mac))
        print("Total FLOPs:{}".format(self.flops))

    def get_mac_layer(self):
        return self.mac_layer

    def get_exit_result(self):
        for layer in range(self.num_layers):
            print(self.exit[layer])
            total = 0
            for item in self.exit[layer].values():
                total += item
            print("Total exit: {}".format(total))

    def get_exclude_result(self):
        for layer in range(self.num_layers):
            print(self.exclude[layer])
