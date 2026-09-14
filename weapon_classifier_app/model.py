import torch
from torch import nn
from torchvision.models import resnet18



# residual block ===========================
class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()

    self.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = nn.Sequential()

    if stride != 1 or in_channels != out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

  def forward(self, x):

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += self.shortcut(x)
    out = self.relu(out)

    return out

# ResNet-18 architecture ========================
class ResNet18(nn.Module):
  def __init__(self,
             color_channels: int= 3,
             num_classes: int= 3):
    super().__init__()

    self.in_channels = 16

    self.conv1 = nn.Conv2d(
        in_channels=color_channels,
        out_channels=16,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    self.bn1 = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(
        kernel_size=3,
        stride=2,
        padding=1
    )

    self.layer1 = self._make_layer(
        block=BasicBlock,
        out_channels=16,
        num_blocks=2,
        stride=1
    )

    self.layer2 = self._make_layer(
        block=BasicBlock,
        out_channels= 32,
        num_blocks=2,
        stride=2
    )

    self.layer3 = self._make_layer(
        block=BasicBlock,
        out_channels= 64 ,
        num_blocks=2,
        stride=2
    )

    self.layer4 = self._make_layer(
        block=BasicBlock,
        out_channels= 128,
        num_blocks=2,
        stride=2
    )

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    self.classifier = nn.Linear(
        in_features= 128 ,
        out_features=num_classes
    )

  def _make_layer(self, block, out_channels, num_blocks, stride):
    layers = []

    layers.append(
        block( self.in_channels, out_channels, stride)
    )

    self.in_channels = out_channels

    for b in range(1, num_blocks):
      layers.append(
          block(self.in_channels,out_channels)
      )

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
    x = torch.flatten(x, 1)

    x = self.classifier(x)

    return x



def load_model(model_path, device):
    model = ResNet18(color_channels=3,
                          num_classes=3)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model