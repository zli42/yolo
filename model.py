import torch
from torch import nn


def basic_block(in_channels, out_channels, kernel_size, stride):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.01),
    )


class ResBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        half_planes = planes // 2
        self.seq = nn.Sequential(
            basic_block(planes, half_planes, 1, 1),
            basic_block(half_planes, planes, 3, 1),
        )

    def forward(self, x):
        identity = x
        y = self.seq(x)
        y += identity
        return y


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes = 32
        self.stem = basic_block(3, self.planes, 3, 1)
        self.layer0 = self._make_layer(1)
        self.layer1 = self._make_layer(2)
        self.layer2 = self._make_layer(8)
        self.layer3 = self._make_layer(8)
        self.layer4 = self._make_layer(4)

    def _make_layer(self, blocks):
        double_planes = self.planes * 2
        seq = nn.Sequential()
        seq.append(basic_block(self.planes, double_planes, 3, 2))
        for _ in range(blocks):
            seq.append(ResBlock(double_planes))
        self.planes = double_planes
        return seq

    def forward(self, x):
        x = self.stem(x)
        x = self.layer0(x)
        x = self.layer1(x)
        y0 = self.layer2(x)
        y1 = self.layer3(y0)
        y2 = self.layer4(y1)
        return y0, y1, y2


class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes = 1024
        self.topdown_planes = 0
        self.lateral_layer0 = self._lateral_layer(self.planes + self.topdown_planes)
        self.topdown_layer0 = self._topdown_layer()
        self.lateral_layer1 = self._lateral_layer(self.planes + self.topdown_planes)
        self.topdown_layer1 = self._topdown_layer()
        self.lateral_layer2 = self._lateral_layer(self.planes + self.topdown_planes)

    def _lateral_layer(self, planes):
        half_planes = self.planes // 2
        seq = nn.Sequential(
            basic_block(planes, half_planes, 1, 1),
            basic_block(half_planes, self.planes, 3, 1),
            basic_block(self.planes, half_planes, 1, 1),
            basic_block(half_planes, self.planes, 3, 1),
            basic_block(self.planes, half_planes, 1, 1),
        )
        self.planes = half_planes
        return seq

    def _topdown_layer(self):
        half_planes = self.planes // 2
        seq = nn.Sequential(
            basic_block(self.planes, half_planes, 1, 1),
            nn.Upsample(scale_factor=2),
        )
        self.topdown_planes = half_planes
        return seq

    def forward(self, x2, x1, x0):
        y0 = self.lateral_layer0(x0)
        h0 = self.topdown_layer0(y0)
        x1 = torch.cat((x1, h0), 1)
        y1 = self.lateral_layer1(x1)
        h1 = self.topdown_layer1(y1)
        x2 = torch.cat((x2, h1), 1)
        y2 = self.lateral_layer2(x2)
        return y0, y1, y2


class Head(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.planes = 512
        self.nodes = num_anchors * (4 + 1 + num_classes)
        self.head0 = self._head_layer()
        self.head1 = self._head_layer()
        self.head2 = self._head_layer()

    def _head_layer(self):
        double_planes = self.planes * 2
        seq = nn.Sequential(
            basic_block(self.planes, double_planes, 3, 1),
            nn.Conv2d(double_planes, self.nodes, 1, 1, 0, bias=True),
        )
        self.planes = self.planes // 2
        return seq

    def forward(self, x0, x1, x2):
        y0 = self.head0(x0)
        y1 = self.head1(x1)
        y2 = self.head2(x2)
        return y0, y1, y2


class YOLO(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.bottleneck = Bottleneck()
        self.head = Head(num_anchors, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0, x1, x2 = self.backbone(x)
        x0, x1, x2 = self.bottleneck(x0, x1, x2)
        y0, y1, y2 = self.head(x0, x1, x2)
        return y0, y1, y2


def test():
    x = torch.rand((2, 3, 416, 416))
    model = YOLO(3, 20)
    y = model(x)
    for each in y:
        print(each.shape)


if __name__ == "__main__":
    test()
