import torch
from .unet import UNet
from .unet_gray import UNet_gray
from .modeling.deeplab import *
from .backbone import resnet
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):

    raise NotImplementedError

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    
    raise NotImplementedError

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model

def unet_rgb(channel=3, num_classes=2):
    print('UNet RGB - Channel: {} Classes: {}'.format(channel, num_classes))
    return UNet(n_channels=channel, n_classes=num_classes)

def unet_gray(channel=1, num_classes=2):
    print('UNet GRAY - Channel: {} Classes: {}'.format(channel, num_classes))
    return UNet_gray(n_channels=channel, n_classes=num_classes)

# Not Implemented Error
def unet_pt(channel=1, num_classes=2):
    print('UNet Pretrained - Channel: {} Classes: {}'.format(channel, num_classes))
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=1, out_channels=2, init_features=32, pretrained=True)

def deeplab(channel=1, num_classes=2):
    print('DeepLab - Channel: {} Classes: {}'.format(channel, num_classes))
    return DeepLab(num_classes=num_classes, backbone='resnet', output_stride=16,
                    sync_bn=True, freeze_bn=False)

def deeplabv3_resnet50(channel=1, num_classes=2, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3_resnet101(channel=1, num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50(channel=1, num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(channel=1, num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
