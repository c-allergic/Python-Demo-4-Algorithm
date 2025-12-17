# Resnet Model
import torch
import torch.nn as nn

class block(nn.Module):
    expansion = 4  
    
    def __init__(self,in_channels, planes, stride=1, identity_downsample=None):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 =nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self,x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self._make_layer(block, layers[0], planes = 64, stride = 1)
        self.layer2 = self._make_layer(block, layers[1], planes = 128, stride = 2)
        self.layer3 = self._make_layer(block, layers[2], planes = 256, stride = 2)
        self.layer4 = self._make_layer(block, layers[3], planes = 512, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def _make_layer(self, block, num_residual_blocks, planes, stride=1):
        layers = []
        identity_downsample = None
        
        if stride != 1 or self.in_channels != planes * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
            
        layers.append(block(self.in_channels, planes, stride, identity_downsample))
        self.in_channels = planes * block.expansion
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, planes, stride=1))
        
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
        x = self.fc(x)
        return x
    
    
def ResNet50(image_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], image_channels, num_classes)

def ResNet101(image_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], image_channels, num_classes)

def ResNet152(image_channels=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], image_channels, num_classes)

def ResNet200(image_channels=3, num_classes=1000):
    return ResNet(block, [3, 24, 36, 3], image_channels, num_classes)

def test():
    model = ResNet50(image_channels=3, num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
    
# if __name__ == "__main__":
#     test()