# Optional list of dependencies required by the package
dependencies = ["torch"]

from vision.nets import ConvNet, CNN7, ResNet18, ResNet50
from vision.convmed import ConvMed, ConvMedBig
from vision.wide_resnet_bn import wide_resnet_8

def resnet18(pretrained=False, **kwargs):
    model = ResNet18(**kwargs)
    if pretrained:
        checkpoint = f'https://github.com/cestwc/models/raw/main/pretrained/{pretrained}.pt'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
    return model
