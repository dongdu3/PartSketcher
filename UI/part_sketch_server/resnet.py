import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
from common import normalize_imagenet
import math


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super(ConvEncoder, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super(Resnet18, self).__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


############################ modified reset18 for image input with different dim ############################
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ModifiedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_dim=6):
        self.inplanes = 64
        super(ModifiedResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ModifiedResnet18(in_dim=6, f_dim=256):
    """Constructs a Modified ResNet-18 model.
    Args:
        in_dim: dim of input image
        f_dim: dim of output feature
    """
    model = ModifiedResNet(BasicBlock, [2, 2, 2, 2], in_dim=in_dim, num_classes=f_dim)

    return model

#####################################################################################################################

class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super(Resnet34, self).__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super(Resnet50, self).__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super(Resnet101, self).__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
