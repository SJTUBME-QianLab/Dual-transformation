import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch
from model.utils import Normalize
from torch.nn.parameter import Parameter
from model.se_resnet import se_resnet_18

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
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

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        feature = self.avgpool(x)
        x = feature.reshape(feature.size(0), -1)
        x = self.classifier(x)

        return x, feature


class one_stream_resnet(nn.Module):
    def __init__(self, model='resnet18', block=BasicBlock, num_classes=1000, pretrained=True):
        super(one_stream_resnet, self).__init__()
        if model == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
        elif model == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            block = Bottleneck
        elif model == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
            block = Bottleneck
        elif model == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
        else:
            backbone = resnet18(pretrained=pretrained)

        self.layer = nn.Sequential(*list(backbone.children())[:-1])  # Remove fc

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion * 2, 256),
            nn.Linear(256, num_classes))
        self.l2norm = Normalize(2)

    def forward(self,  x_a, x_v):

        feature_a = self.layer(x_a)  # [64, 512, 1, 1]
        feature_v = self.layer(x_v)
        feature_a = feature_a.reshape(feature_a.size(0), -1)  # [64, 512]
        feature_v = feature_v.reshape(feature_v.size(0), -1)
        out_feature = torch.cat([feature_a, feature_v], dim=1)  # [64, 1024]

        out = self.classifier(out_feature)
        # return out, torch.cat([feature_a, feature_v], dim=1), out_a, out_v
        return out, self.l2norm(out_feature, dim=1)


class two_stream_resnet(nn.Module):
    def __init__(self, model='resnet18', block=BasicBlock, num_classes=1000, pretrained=True):
        super(two_stream_resnet, self).__init__()
        if model == 'resnet34':
            backbone1 = resnet34(pretrained=pretrained)
            backbone2 = resnet34(pretrained=pretrained)
        elif model == 'resnet50':
            backbone1 = resnet50(pretrained=pretrained)
            backbone2 = resnet50(pretrained=pretrained)
            block = Bottleneck
        elif model == 'resnet101':
            backbone1 = resnet101(pretrained=pretrained)
            backbone2 = resnet101(pretrained=pretrained)
            block = Bottleneck
        elif model == 'resnet18':
            backbone1 = resnet18(pretrained=pretrained)
            backbone2 = resnet18(pretrained=pretrained)
        else:
            backbone1 = resnet18(pretrained=pretrained)
            backbone2 = resnet18(pretrained=pretrained)

        self.layer1_a = nn.Sequential(*list(backbone1.children())[:-1])  # Remove fc
        self.layer1_v = nn.Sequential(*list(backbone1.children())[:-1])  # Remove fc
        self.layer2_a = nn.Sequential(*list(backbone2.children())[:-1])  # Remove fc
        self.layer2_v = nn.Sequential(*list(backbone2.children())[:-1])  # Remove fc

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion * 2, 256),
            nn.Linear(256, num_classes))
        self.l2norm = Normalize(2)
        L = 128
        self.w = Parameter(torch.Tensor(1, L))
        self.v = Parameter(torch.Tensor(L, 512 * block.expansion * 2))
        nn.init.kaiming_uniform_(self.w, a=0.1)  # leakyrelu 0.1
        nn.init.kaiming_uniform_(self.v, a=0.1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * block.expansion * 2, num_classes))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self,  x1_a, x1_v, x2_a, x2_v):

        feature1_a = self.layer1_a(x1_a)  # [64, 512, 1, 1]
        feature1_v = self.layer1_v(x1_v)
        feature1_a = feature1_a.reshape(feature1_a.size(0), -1)  # [64, 512]
        feature1_v = feature1_v.reshape(feature1_v.size(0), -1)
        out_feature1 = torch.cat([feature1_a, feature1_v], dim=1)  # [64, 1024]
        atime1 = torch.matmul(self.w, torch.tanh(torch.matmul(self.v, out_feature1.unsqueeze(2))))

        feature2_a = self.layer2_a(x2_a)
        feature2_v = self.layer2_v(x2_v)
        feature2_a = feature2_a.reshape(feature2_a.size(0), -1)
        feature2_v = feature2_v.reshape(feature2_v.size(0), -1)
        out_feature2 = torch.cat([feature2_a, feature2_v], dim=1)
        atime2 = torch.matmul(self.w, torch.tanh(torch.matmul(self.v, out_feature2.unsqueeze(2))))

        out_feature = torch.cat([out_feature1.unsqueeze(1), out_feature2.unsqueeze(1)], dim=1)  # [64, 2, 1024] 两个支流的特征
        a = torch.cat([atime1.reshape(-1, 1), atime2.reshape(-1, 1)], dim=1)  # [64, 2]
        a = self.Softmax(a)
        feature = torch.mul(a.unsqueeze(2), out_feature)  # [64, 2, 1024]
        feature = torch.sum(feature, dim=1)  # [64, 1024]  融合之后的特征

        out = self.classifier(feature)
        # return out, torch.cat([feature_a, feature_v], dim=1), out_a, out_v
        return out, self.l2norm(feature, dim=1), self.l2norm(out_feature, dim=2)


class three_stream_resnet(nn.Module):
    def __init__(self, model='resnet18', block=BasicBlock, num_classes=1000, pretrained=True):
        super(three_stream_resnet, self).__init__()
        if model == 'resnet34':
            backbone1 = resnet34(pretrained=pretrained)
            backbone2 = resnet34(pretrained=pretrained)
            backbone3 = resnet34(pretrained=pretrained)
        elif model == 'resnet50':
            backbone1 = resnet50(pretrained=pretrained)
            backbone2 = resnet50(pretrained=pretrained)
            backbone3 = resnet50(pretrained=pretrained)
            block = Bottleneck
        elif model == 'resnet101':
            backbone1 = resnet101(pretrained=pretrained)
            backbone2 = resnet101(pretrained=pretrained)
            backbone3 = resnet101(pretrained=pretrained)
            block = Bottleneck
        elif model == 'resnet18':
            backbone1 = resnet18(pretrained=pretrained)
            backbone2 = resnet18(pretrained=pretrained)
            backbone3 = resnet18(pretrained=pretrained)
        else:
            backbone1 = resnet18(pretrained=pretrained)
            backbone2 = resnet18(pretrained=pretrained)
            backbone3 = resnet18(pretrained=pretrained)

        self.layer1 = nn.Sequential(*list(backbone1.children())[:-1])  # Remove fc
        self.layer2 = nn.Sequential(*list(backbone2.children())[:-1])  # Remove fc
        self.layer3 = nn.Sequential(*list(backbone3.children())[:-1])  # Remove fc

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion * 2, 256),
            nn.Linear(256, num_classes))

        self.l2norm = Normalize(2)
        L = 128
        self.w = Parameter(torch.Tensor(1, L))
        self.v = Parameter(torch.Tensor(L, 512 * block.expansion * 2))
        nn.init.kaiming_uniform_(self.w, a=0.1)  # leakyrelu 0.1
        nn.init.kaiming_uniform_(self.v, a=0.1)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * block.expansion * 2, num_classes))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self,  x1_a, x1_v, x2_a, x2_v, x3_a, x3_v):

        feature1_a = self.layer1(x1_a)  # [64, 512, 1, 1]
        feature1_v = self.layer1(x1_v)
        feature1_a = feature1_a.reshape(feature1_a.size(0), -1)  # [64, 512]
        feature1_v = feature1_v.reshape(feature1_v.size(0), -1)
        out_feature1 = torch.cat([feature1_a, feature1_v], dim=1)  # [64, 1024]
        atime1 = torch.matmul(self.w, torch.tanh(torch.matmul(self.v, out_feature1.unsqueeze(2))))

        feature2_a = self.layer2(x2_a)
        feature2_v = self.layer2(x2_v)
        feature2_a = feature2_a.reshape(feature2_a.size(0), -1)
        feature2_v = feature2_v.reshape(feature2_v.size(0), -1)
        out_feature2 = torch.cat([feature2_a, feature2_v], dim=1)
        atime2 = torch.matmul(self.w, torch.tanh(torch.matmul(self.v, out_feature2.unsqueeze(2))))

        feature3_a = self.layer3(x3_a)
        feature3_v = self.layer3(x3_v)
        feature3_a = feature3_a.reshape(feature3_a.size(0), -1)
        feature3_v = feature3_v.reshape(feature3_v.size(0), -1)
        out_feature3 = torch.cat([feature3_a, feature3_v], dim=1)
        atime3 = torch.matmul(self.w, torch.tanh(torch.matmul(self.v, out_feature3.unsqueeze(2))))

        out_feature = torch.cat([out_feature1.unsqueeze(1), out_feature2.unsqueeze(1), out_feature3.unsqueeze(1)], dim=1)  # [64, 3, 1024] 两个支流的特征
        a = torch.cat([atime1.reshape(-1, 1), atime2.reshape(-1, 1), atime3.reshape(-1, 1)], dim=1)  # [64, 3]
        a = self.Softmax(a)
        feature = torch.mul(a.unsqueeze(2), out_feature)  # [64, 3, 1024]
        feature = torch.sum(feature, dim=1)  # [64, 1024]  融合之后的特征

        out = self.classifier(feature)
        # return out, torch.cat([feature_a, feature_v], dim=1), out_a, out_v
        return out, self.l2norm(feature, dim=1), self.l2norm(out_feature, dim=2)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('pretrained is True')
        pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 导入相同key的权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
