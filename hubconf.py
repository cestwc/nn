# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from vision.nets import MNISTCNN, CNN7, ResNet18, ResNet50
from vision.convmed import ConvMed, ConvMedBig
from vision.wide_resnet_bn import wide_resnet_8 as _wide_resnet_8

def resnet18(pretrained=False, **kwargs):
    model = ResNet18(**kwargs)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model

def convnet_mnist(pretrained=False, **kwargs):
    model = ConvNet(**kwargs)
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
    model = _wide_resnet_8(**kwargs)
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
