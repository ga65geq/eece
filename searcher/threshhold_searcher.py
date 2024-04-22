import torch
import copy
from data_prep.data_praperation import DataPrep
from model_trainer.trainer import LitModel

class Searcher:
    def __init__(self, args):
        self.thresh_hold = [0 for i in range(args.num_exit_points)]
        self.device = args.device_num[0]
        self.num_exit_points = args.num_exit_points
        model = LitModel.load_from_checkpoint(args.ckpt_path, args=args)
        model = model.model
        model = model.eval()
        model.cuda(self.device)
        self.model = model

        dataprep = DataPrep(args.data_cache_dir, args.dataset)
        train_loader, val_loader, test_loader = dataprep.get_dataloader(batch_size=args.batch_size)
        self.dataloader = val_loader

        self.tolerance = args.tolerance
        self.num_exit_points = args.num_exit_points

        self.start_acc = 0
        self.goal_acc = 0
        self.start = args.start_thresh
        self.step = args.step
        self.args = args

        acc = self._run_model()
        self.start_acc = acc
        self.goal_acc = acc - self.tolerance
        self.layer_drop = self._get_layer_drop()
        print(self.layer_drop)

    def _run_model(self):
        correct = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                data, target = data.cuda(self.device), target.cuda(self.device)
                output = self.model(data)
                idx = torch.sum(self.model.exit_strategy.res, 1) == 0
                self.model.exit_strategy.res[idx, :] = 1
                pred = (output[0]*self.model.exit_strategy.res).argmax(dim=1, keepdim=True)
                if self.args.snet:
                    self.model.exit_strategy.clear()
                correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(self.dataloader.dataset)
            print("accuracy: {}".format(acc))
        return acc

    def _get_layer_drop(self):
        mac_layer = self.model.statistic.get_mac_layer()
        total = 0
        layer_drop = []
        # for i in range(1, self.num_exit_points + 1):
        #     total += mac_layer[i]
        for i in range(self.num_exit_points):
            drop = mac_layer[i]
            layer_drop.append(drop)
        return layer_drop

    def search(self):
        if self.args.strategy_name == "LargeN":
            # return self._iter_search()
            return self._large_n_search()
        elif self.args.strategy_name == "CompareN":
            # return self._iter_search()
            return self._compare_n_search()

    def _large_n_search(self):
        layers = sorted(range(len(self.layer_drop)), key=lambda k: self.layer_drop[k], reverse=True)
        for current_layer in layers:
            print("Current layer: {}".format(current_layer))
            acc = self.start_acc
            self.thresh_hold[current_layer] = self.start
            self.model.exit_strategy.change_thresh_hold(self.thresh_hold)
            while acc > self.goal_acc and 1 >= self.thresh_hold[current_layer] > 0:
                print(self.model.exit_strategy.thresh_hold)
                acc = self._run_model()
                self.thresh_hold[current_layer] += self.step
                self.model.exit_strategy.change_thresh_hold(self.thresh_hold)
            self.thresh_hold[current_layer] -= (2*self.step)
            self.thresh_hold[current_layer] = max(0.01, self.thresh_hold[current_layer])
            self.thresh_hold[current_layer] = min(0.99, self.thresh_hold[current_layer])
        return self.thresh_hold

    def _compare_n_search(self):
        # layers = list(range(self.num_exit_points))
        # layers.reverse()
        layers = sorted(range(len(self.layer_drop)), key=lambda k: self.layer_drop[k], reverse=True)
        for current_layer in layers:
            print("Current layer: {}".format(current_layer))
            acc = self.start_acc
            self.thresh_hold[current_layer] = self.start
            self.model.exit_strategy.change_thresh_hold(self.thresh_hold)
            while acc > self.goal_acc and 1 >= self.thresh_hold[current_layer] > 0:
                print(self.model.exit_strategy.thresh_hold)
                acc = self._run_model()
                self.thresh_hold[current_layer] -= self.step
                self.model.exit_strategy.change_thresh_hold(self.thresh_hold)
            self.thresh_hold[current_layer] += (2*self.step)
            self.thresh_hold[current_layer] = max(0.01, self.thresh_hold[current_layer])
            self.thresh_hold[current_layer] = min(0.99, self.thresh_hold[current_layer])
        return self.thresh_hold

    def _iter_search(self):
        layers = [i for i in range(len(self.layer_drop))]
        layers.reverse()
        for current_layer in layers:
            best_acc = 0
            acc = 1
            th=0.01
            tmp_threshold = copy.deepcopy(self.thresh_hold)
            tmp_threshold[current_layer] = th
            self.model.exit_strategy.change_thresh_hold(tmp_threshold)
            print("Current layer: {}".format(current_layer))
            while th < 1 and acc > self.start_acc-0.1:
                print(th)
                acc = self._run_model()
                if acc > best_acc:
                    best_acc = acc
                    self.thresh_hold[current_layer] = th
                th += 0.01
                tmp_threshold[current_layer] = th
            print(self.thresh_hold)



