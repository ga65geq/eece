import torch.optim
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl

from model.model_factory import model_fectory


class MyLoss(nn.Module):
    def __init__(self, num_classes, pos_weight, num_exit_points, args):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        pos_weight = torch.ones(num_classes) * pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight)
        self.num_exit_points = num_exit_points
        self.factor_decay = False
        if hasattr(args, "factor_decay"):
            self.factor_decay = args.factor_decay
            self.count = 1
            self.tmp_count = 0

    def forward(self, output, target):
        x, y = output
        btarget = torch.zeros_like(x)
        btarget = btarget.scatter(1, target.view(-1, 1), 1)

        loss = self.cross_entropy(x, target)
        bce = None
        for idx, item in enumerate(y):
            if bce is None:
                bce = 1 / (self.num_exit_points - idx + 1) * self.bce(item, btarget)
            else:
                bce += 1 / (self.num_exit_points - idx + 1) * self.bce(item, btarget)
        if self.factor_decay:
            self.tmp_count += 1
            if self.tmp_count == 30000:
                self.count *= 2
            if self.tmp_count == 60000:
                self.count *= 2
            if self.tmp_count == 80000:
                self.count *= 2

        return loss + bce / self.count


class MyLoss1(nn.Module):
    def __init__(self, num_exit_points, args):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_exit_points = num_exit_points

    def forward(self, output, target):
        x, y = output

        loss = self.cross_entropy(x, target)
        bce = None
        for idx, item in enumerate(y):
            if bce is None:
                bce = 1 / (self.num_exit_points - idx + 1) * self.cross_entropy(item, target)
            else:
                bce += 1 / (self.num_exit_points - idx + 1) * self.cross_entropy(item, target)

        return loss + bce


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = model_fectory(args)
        self.snet = args.snet

        if args.snet:
            if args.activation == "sigmoid":
                self.loss = MyLoss(args.num_classes, args.pos_weight, args.num_exit_points, args)
            else:
                self.loss = MyLoss1(args.num_exit_points, args)
        else:
            self.loss = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        if self.snet:
            loss = self.loss(output, y)
        else:
            loss = self.loss(output[0], y)
        pred = output[0].max(dim=1, keepdim=True)[1]
        self.log("train_loss", loss.item())
        self.train_acc(pred, y.view_as(pred))
        return loss

    def on_train_epoch_end(self) -> None:
        self.log('train_acc', self.train_acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        if self.snet:
            loss = self.loss(output, y)
        else:
            loss = self.loss(output[0], y)
        pred = output[0].max(dim=1, keepdim=True)[1]
        self.log("val_loss", loss.item())
        self.val_acc(pred, y.view_as(pred))
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        if self.snet:
            loss = self.loss(output, y)
        else:
            loss = self.loss(output[0], y)
        pred = output[0].max(dim=1, keepdim=True)[1]
        self.test_acc(pred, y.view_as(pred))
        return loss

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        if self.args.architecture == 'Alexnet':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        return [optimizer], [scheduler]
