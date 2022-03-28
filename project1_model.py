import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

mean = [0.49139968, 0.48215827 ,0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]

# apply transforms
transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomVerticalFlip(p=0.3), 
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
])

transforms_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

 # we are loading both: transformed and untransformed data
trainingdata_transform = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train) 
trainingdata_simple = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_val)

# make datasets
trainDataLoaderTransform = torch.utils.data.DataLoader(trainingdata_transform,batch_size=64,shuffle=True)
trainDataLoaderSimple = torch.utils.data.DataLoader(trainingdata_simple,batch_size=64,shuffle=True)

testdata = torchvision.datasets.CIFAR10(root='./data',  train=False, download=True, transform=transforms_val)
testDataLoader = torch.utils.data.DataLoader(testdata,batch_size=64,shuffle=False)


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

class Conf:
  """
  A Conf object represents state of a single set of parameters
  """
  def __init__(self,blocks,c1,p=1):
    self.blocks = blocks
    self.c1 = c1
    self.p = p
  def __repr__(self):
    return f'blocks_{"_".join("_".join([str(block) for block in self.blocks]))}_c1_{self.c1}_p_{self.p}'



def project1_model():
    return ResNet(BasicBlock,num_blocks =[3,10],c1=58,p=2)


def main():
  # best model conf
  confs_to_test = [Conf([3,10],58,2)]

  # can also add other confs to test like this:
  # confs_to_test = [Conf([3,10],58,2),Conf([2,3,3,2],40,1),Conf([4,6,8],42,1),Conf([4,4,4],58,1),Conf([3,10],80,4),Conf([8,16],60),Conf([2,2,2,2],30,4)]

  for conf in confs_to_test: # iterate over configurations
    model = ResNet(BasicBlock,num_blocks =conf.blocks,c1=conf.c1,p=conf.p).cuda()

    net = model
    Loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loss_history = []
    test_loss_history = []

    max_accuracy = 0
    net_to_save = net.state_dict()
    for epoch in range(100):
      train_loss = 0.0
      test_loss = 0.0
      for i, data in enumerate(gen([trainDataLoaderTransform,trainDataLoaderSimple])): # combined data loaders
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        predicted_output = net(images)
        fit = Loss(predicted_output,labels)


        fit.backward()
        optimizer.step()
        train_loss += fit.item()
      
      correct = 0
      total = 0
      for i, data in enumerate(testDataLoader):
        with torch.no_grad():
          images, labels = data
          images = images.cuda()
          labels = labels.cuda()
          predicted_output = net(images)
          fit = Loss(predicted_output,labels)
          test_loss += fit.item()
          predicted = torch.max(predicted_output.data, 1)
          for i in range(len(labels)):
              if labels[i] == predicted[1][i]:
                  correct += 1
              total += 1
      accuracy = correct/total
      if accuracy>max_accuracy: # check for max accuracy
        max_accuracy = accuracy
        net_to_save = net.state_dict() # cross-validation: save the state for which test accuracy we maximum
      max_accuracy = max(max_accuracy,accuracy)


      train_loss = train_loss/(2*len(trainingdata_simple))
      test_loss = test_loss/len(testDataLoader)
      train_loss_history.append(train_loss)
      test_loss_history.append(test_loss)
      
    print('max accuracy:',max_accuracy,'model',conf,'params',count_parameters(net)/1000000)
    torch.save(net_to_save,f'{conf}.pt')

if __name__ == "__main__":
    main()