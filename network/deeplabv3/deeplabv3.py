import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabV3(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                sync_bn=True, freeze_bn=False):
        super(DeepLabV3, self).__init__()

        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 
        #                        'deeplabv3_resnet50', pretrained=True)
        #self.model = deeplabv3_resnet50(pretrained=True)
        #self.out = nn.Conv2d(21, 2, kernel_size=(1,1))

        self.net = nn.Sequential(deeplabv3_resnet50(pretrained=True), 
                                nn.Conv2d(21, 2, kernel_size=(1,1)))

    def forward(self, x):

        #x1 = self.model(x)
        #logits = self.out(x1)

        return self.net(x)