import torch
import torch.nn as nn
import torch.nn.functional as F



# a function to combine data loaders
def gen(loaders):
  for loader in loaders:
    for data in loader:
      yield data

# ResNet architecture
class BasicBlock(nn.Module):

  def __init__(self, in_planes, planes,fi, stride=1):
      super(BasicBlock, self).__init__()
      self.conv1 = nn.Conv2d(
          in_planes, planes, kernel_size=fi, stride=stride, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(planes)
      self.conv2 = nn.Conv2d(planes, planes, kernel_size=fi,
                              stride=1, padding=1, bias=False)
      self.bn2 = nn.BatchNorm2d(planes)

      self.shortcut = nn.Sequential()
      if stride != 1 or in_planes != planes:
          self.shortcut = nn.Sequential(
              nn.Conv2d(in_planes, planes,
                        kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(planes)
          )

  def forward(self, x):
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
      out += self.shortcut(x)
      out = F.relu(out)
      return out



class ResNet(nn.Module):
    """
    modified ResNet code: We have made resnet parameterizable so that something like grid search
    can be performed programmatically
    """
    def __init__(self, block, num_blocks,c1,p,f=[3,3,3,3]):
        super(ResNet, self).__init__()
        self.in_planes = c1
        self.p = p

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)

        self.l = len(num_blocks)

        self.layer1 = self._make_layer(block, c1, num_blocks[0],f[0], stride=1)

        if self.l>1:
          self.layer2 = self._make_layer(block, 2*c1, num_blocks[1],f[1], stride=2)

        if self.l>2:
          self.layer3 = self._make_layer(block, 4*c1, num_blocks[2],f[2], stride=2)

        if self.l>3:
          self.layer4 = self._make_layer(block, 8*c1, num_blocks[3],f[3], stride=2)

        if self.l>4:
          self.layer5 = self._make_layer(block, 16*c1, num_blocks[4],f[4], stride=2)

        last_in_size = (2**(self.l-1))*c1
        last_dim = 64//(2**self.l)
        outsize = (last_dim//(self.p))**2 * last_in_size # calculating out size based on input params
        self.linear = nn.Linear(outsize, 10)

    def _make_layer(self, block, planes, num_blocks,fi, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,fi,stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        if self.l>1:
          out = self.layer2(out)
        if self.l>2:
          out = self.layer3(out)
        if self.l>3:
          out = self.layer4(out)
        if self.l>4:
          out = self.layer5(out)
        out = F.avg_pool2d(out, self.p)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock,num_blocks =[3,10],c1=58,p=2)
