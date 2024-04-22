import torch
from functools import partial


def create_strategy(args):
    if args.activation == "sigmoid":
        activation = torch.sigmoid
    else:
        activation = partial(torch.softmax, dim=1)

    if args.strategy_name == "":
        strategy = None

    elif args.strategy_name == "LargeN":
        if hasattr(args, "thresh_hold"):
            thresh_hold = args.thresh_hold
        else:
            thresh_hold = []
        if hasattr(args, "search"):
            strategy = LargeN(thresh_hold, activation, args.device_name, args.search, args.num_classes)
        else:
            strategy = LargeN(thresh_hold, activation, args.device_name, False, args.num_classes)

    elif args.strategy_name == "CompareN":
        if hasattr(args, "thresh_hold"):
            thresh_hold = args.thresh_hold
        else:
            thresh_hold = []
        if hasattr(args, "search"):
            strategy = CompareN(thresh_hold, activation, args.device_name, args.search, args.num_classes)
        else:
            strategy = CompareN(thresh_hold, activation, args.device_name, False, args.num_classes)
    else:
        raise NotImplementedError
    return strategy


class Strategy:
    def __init__(self, thresh_hold: list, activation, device, search, num_classes):
        self.reset = True
        if search:
            self.res = None
            self.func = self._search
        else:
            self.res = torch.ones(1, num_classes, device=torch.device(device), dtype=bool)
            self.func = self._inference
        self.thresh_hold = thresh_hold
        self.current = 0
        self.activation = activation

    def __call__(self, y):
        self.func(y)

    def clear(self):
        self.reset = True

    def change_thresh_hold(self, thresh_hold):
        self.thresh_hold = thresh_hold

    def _search(self, y):
        pass

    def _inference(self, y):
        pass


class CompareN(Strategy):
    def _search(self, y):
        if self.reset:
            self.reset = False
            self.res = torch.ones_like(y, dtype=bool)
            self.current = 0
        if self.current < len(self.thresh_hold):
            tmp = self.activation(y)
            self.res[torch.sum(self.res, dim=1) != 1] = \
                tmp[torch.sum(self.res, dim=1) != 1] > self.thresh_hold[self.current]
        self.current += 1

    def _inference(self, y):
        if self.reset:
            self.reset = False
            self.res[:] = True
            self.current = 0
        tmp = self.activation(y)
        self.res = tmp > self.thresh_hold[self.current]
        self.current += 1


class LargeN(Strategy):
    def _search(self, y):
        if self.reset:
            self.reset = False
            self.res = torch.ones_like(y, dtype=bool)
            self.current = 0
        if self.current < len(self.thresh_hold):
            y = self.activation(y)
            max_tmp, idx = y.max(dim=1, keepdim=True)
            unfinished = torch.sum(self.res, dim=1) != 1
            self.res[unfinished, idx[unfinished]] = True
            tmp = y[unfinished]
            standard = max_tmp[unfinished] * self.thresh_hold[self.current]
            self.res[unfinished] = (tmp > standard) * self.res[unfinished]
        self.current += 1

    def _inference(self, y):
        if self.reset:
            self.reset = False
            self.res[:] = True
            self.current = 0
        tmp_y = self.activation(y)
        max_tmp, idx = tmp_y.max(dim=1, keepdim=True)
        self.res[:, idx] = True
        # standard = max_tmp * self.thresh_hold[self.current]
        self.res = torch.logical_and((tmp_y > (max_tmp * self.thresh_hold[self.current])), self.res)
        self.current += 1
