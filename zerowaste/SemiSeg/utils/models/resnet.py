import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ResNet",
    "resnet50",
    "resnet101",
]

# model_urls = {
#     "resnet34": "/path/to/resnet34.pth",
#     "resnet50": "/zerowaste/SemiSeg/resnet50-0676ba61.pth",
#     "resnet101": "/path/to/resnet101.pth",
# }

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
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

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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
    def __init__(
        self,
        block,
        layers,
        num_classes=1,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, True, True],
        multi_grid=False,
    ):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.stream1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, 
                      dilation=1, padding=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.stream2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, 
                      dilation=2, padding=2),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )        
        self.stream3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, 
                      dilation=4, padding=4),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )        
        self.stream4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, 
                      dilation=8, padding=8),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            multi_grid=multi_grid,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        dilations = [6, 12, 18, 24,]
        self.conv_dilated_6 = nn.Sequential(
            nn.Conv2d(
                2048,
                256,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        self.conv_dilated_12 = nn.Sequential(
            nn.Conv2d(
                2048,
                256,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        self.conv_dilated_18 = nn.Sequential(
            nn.Conv2d(
                2048,
                256,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        self.conv_dilated_24 = nn.Sequential(
            nn.Conv2d(
                2048,
                256,
                kernel_size=3,
                padding=dilations[3],
                dilation=dilations[3],
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_outplanes(self):
        return self.inplanes

    def get_auxplanes(self):
        return self.inplanes // 2

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=False):
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

        grids = [1] * blocks
        if multi_grid:
            grids = [2, 2, 4]

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation * grids[0],
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation * grids[i],
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, inputs):
        resnet_inputs = self.stream1(inputs) + \
            self.stream2(inputs) + \
            self.stream3(inputs) + \
            self.stream4(inputs)
        
        x = self.relu(self.bn1(self.conv1(resnet_inputs)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        backbone = self.avgpool(x4)
        conv_dilated_6 = self.conv_dilated_6(backbone)
        conv_dilated_12 = self.conv_dilated_12(backbone)
        conv_dilated_18 = self.conv_dilated_18(backbone)
        conv_dilated_24 = self.conv_dilated_24(backbone)
        classifier = self.classifier(
            conv_dilated_6 + conv_dilated_12 + conv_dilated_18 + conv_dilated_24)
        _, _, h, w = inputs.size()
        preds = F.interpolate(
            classifier, size=(h, w), mode="bilinear", align_corners=True)
        return preds

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model_url = model_urls["resnet50"]
    #     state_dict = torch.load(model_url)

    #     missing_keys, unexpected_keys = \
    #         model.load_state_dict(state_dict, strict=False)
    #     print(
    #         f"[Info] Load ImageNet pretrain from '{model_url}'",
    #         "\nmissing_keys: ",
    #         missing_keys,
    #         "\nunexpected_keys: ",
    #         unexpected_keys,
    #     )
    return model

def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # if pretrained:
    #     model_url = model_urls["resnet101"]
    #     state_dict = torch.load(model_url)

    #     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    #     print(
    #         f"[Info] Load ImageNet pretrain from '{model_url}'",
    #         "\nmissing_keys: ",
    #         missing_keys,
    #         "\nunexpected_keys: ",
    #         unexpected_keys,
    #     )
    return model

if __name__ == "__main__":
    seg_model = resnet101()
    inputs = torch.rand((1, 3, 513, 513))
    outputs = seg_model(inputs)
    print(1)