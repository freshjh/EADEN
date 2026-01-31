import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from .attention import ChannelWiseTransformerBlock, HighFreqLearning
from CLIP.model import ModifiedResNet

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}





def channel_shuffle(x1, x2):
    x = torch.cat((x1, x2), 1)
    batch_size, channels, height, width = x.data.size()
    groups = 2
    assert channels % groups == 0
    group_channels = channels // groups

    out = x.reshape(batch_size, group_channels, groups, height, width)
    out = out.permute(0, 2, 1, 3, 4)
    out = out.reshape(batch_size, channels, height, width)

    y1, y2 = torch.chunk(out, 2, dim=1)
    return y1, y2







def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        # dire = self.conv1(torch.concat([dire, img], dim=1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # dire = self.avgpool(dire)
        # dire = dire.view(dire.size(0), -1)
        # dire = self.fc(dire)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:

        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model



class channel_mix_Trans(nn.Module):
    def __init__(self, in_channels):
        super(channel_mix_Trans, self).__init__()
        
        self.CT1 = ChannelWiseTransformerBlock(in_channels)
        self.CT2 = ChannelWiseTransformerBlock(in_channels)

    def forward(self, x1, x2):

        y1, y2 = channel_shuffle(x1, x2)

        y1 = self.CT1(y1)
        y2 = self.CT2(y2)
        
        return x1+y1, x2+y2



class DualNet(nn.Module):
    def __init__(self):
        super().__init__()

        # feature encoder
        self.net_img = ModifiedResNet((3,4,6,3), 1024, 32, width=64)
        CLIP_weight = torch.load('/opt/data/private/Projects/AIGCDet/DIRE-Variants/DIRE-ChannelShuffle/DIRE-Channel-Spatial/CLIP-R50.pth', map_location = 'cpu')
        self.net_img.load_state_dict(CLIP_weight, strict=False)
        del CLIP_weight



        self.net_dire = resnet50(pretrained=True)
       
        self.EAS = channel_mix_Trans(256)

        self.HDL = HighFreqLearning(256)



        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # clasifier 
        self.fc = nn.Linear(2048*2, 1)

        nn.init.normal_(self.fc.weight.data, 0.0, 0.02)


    def forward(self, img, dire):
        
        def stem(x):
            x = self.net_img.relu1(self.net_img.bn1(self.net_img.conv1(x)))
            x = self.net_img.relu2(self.net_img.bn2(self.net_img.conv2(x)))
            x = self.net_img.relu3(self.net_img.bn3(self.net_img.conv3(x)))
            x = self.net_img.avgpool(x)
            return x
        
        img = stem(img)
        
        
        dire = self.net_dire.conv1(dire)
        dire = self.net_dire.bn1(dire)
        dire = self.net_dire.relu(dire)
        dire = self.net_dire.maxpool(dire)


        img = self.net_img.layer1(img)
        dire = self.net_dire.layer1(dire)

        img, dire = self.EAS(img, dire)
        img, dire = self.HDL(img, dire)

        img = self.net_img.layer2(img)
        dire = self.net_dire.layer2(dire)


        img = self.net_img.layer3(img)
        dire = self.net_dire.layer3(dire)


        img = self.net_img.layer4(img)
        dire = self.net_dire.layer4(dire)


        img = self.avgpool(img).view(img.size(0), -1)
        dire = self.avgpool(dire).view(dire.size(0), -1)
        
        
        
        feat_fused = torch.cat((img, dire), dim=1)
        
        
        
        
        return self.fc(feat_fused)
        
