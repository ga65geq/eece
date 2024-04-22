import torch

from model.alexnet import AlexNet
from model.vgg import vgg_factory
from model.resnet1 import ResNet, BottleNeck
from model.resnet4imgnet import ResNet as ResNet224
# from model.resnet4imgnet import Bottleneck
from model.MSDNet import MSDNet
from model.mobilenet import MobileNet
from model.inceptionv3 import InceptionV3
from exit_strategy.strategy import create_strategy
from functools import partial
from utils.statistic import Statistic

def model_fectory(args):
    if args.architecture == "Alexnet":
        model = AlexNet(args.num_classes, 0.5, args.snet, args.expansion)
        model.load_pretrained_model(args.pretrained_path)
    elif args.architecture == "Vggsmall":
        if args.snet:
            model = vgg_factory('MS', args.num_classes, args.expansion)
        else:
            model = vgg_factory('M', args.num_classes, args.expansion)
    elif args.architecture == "Vgg16":
        if args.snet:
            model = vgg_factory('BS', args.num_classes, args.expansion)
        else:
            model = vgg_factory('B', args.num_classes, args.expansion)
    elif args.architecture == "ResNet50":
        model = ResNet(BottleNeck, [3, 4, 6, 3], args.num_classes, args.expansion, args.snet)
    elif args.architecture == "ResNet50_imgnet":
        model = ResNet224(BottleNeck, [3, 4, 6, 3], args.num_classes, args.expansion, args.snet)
        pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model.load_pretrained_model(pretrained)
    elif args.architecture == "MSDNet":
        model = MSDNet(args)
    elif args.architecture == "MobileNetV1":
        model = MobileNet(1, args.num_classes, args.expansion, args.snet)
    elif args.architecture == "InceptionV3":
        model = InceptionV3(args.num_classes, args.expansion, args.snet)
    else:
        raise NotImplementedError
    model.set_forward(args.forward_type)
    strategy = create_strategy(args)
    statistic = Statistic(args.num_exit_points, args.num_classes)
    model.set_statistic(statistic)
    model.set_exit_strategy(strategy)
    if args.activation == "sigmoid":
        model.set_exit_activation(torch.sigmoid)
    else:
        model.set_exit_activation(partial(torch.softmax, dim=1))
    return model
