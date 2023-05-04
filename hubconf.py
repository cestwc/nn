# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from vision.nets import MNISTCNN, CNN7, ResNet as ResNet_Zhang, BasicBlock as BasicBlock_Zhang
from vision.wide_resnet_bn import ResNet as WRN_Xu, BasicBlock as BasicBlock_Xu
from vision.convmed import ConvMed, ConvMedBig


def mnistcnn(pretrained=False):
    model = MNISTCNN()
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def resnet18(pretrained=False, **kwargs):
    model = ResNet_Zhang(BasicBlock_Zhang, [2, 2, 2, 2], **kwargs)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def cnn7(pretrained=False, **kwargs):
    model = CNN7(**kwargs)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def wide_resnet_8(pretrained=False, **kwargs):
    model = WRN_Xu(BasicBlock_xu, [1,1,1], widen_factor=8, dense=True, pool=False)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def cifarresnet110(pretrained=False, **kwargs):
    import pytorchcv.model_provider
    model = pytorchcv.model_provider.get_model(f"resnet110_cifar10", pretrained=False)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def cifarwrn16_10(pretrained=False, **kwargs):
    import pytorchcv.model_provider
    model = pytorchcv.model_provider.get_model(f"wrn16_10_cifar100", pretrained=False)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


def convmed(pretrained=False, **kwargs):
    model = ConvMed(**kwargs)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model

def convmedbig(pretrained=False, **kwargs):
    model = ConvMedBig(**kwargs)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model


