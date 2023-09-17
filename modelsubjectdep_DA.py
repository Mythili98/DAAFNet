import torch
import torch.nn as nn
from torch.autograd import Function

# Model architechture
# Convolution layer : 32 ﬁlters size of (3 × 3
# or 5 × 5 or 7 × 7)
# Residual block 1 : 32 ﬁlters size of (3 × 3
# or 5 × 5 or 7 × 7)
# Residual block 2: 64 ﬁlters size of (3 × 3
# or 5 × 5 or 7 × 7)
# Residual block 3: 128 ﬁlters size of (3 × 3
# or 5 × 5 or 7 × 7)
# Dense layer 1
# Dense layer 2
# SoftMax

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride= 1, padding= 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride= 1, padding= 2)
    
    def forward(self,x):
         residual  = torch.clone(x)
         x = self.bn1(x)
         x = self.activation1(x)
         x = self.conv1(x)
         x = self.conv2(self.activation2(self.bn2(x)))
         residual = (residual.unsqueeze(0))
         residual = nn.functional.interpolate(residual, size = [x.shape[1], x.shape[2], x.shape[3]],)
         residual = residual.squeeze(0)
         x += residual
         return x

class LabelClassifier(nn.Module):
    def __init__(self):
        super(LabelClassifier, self).__init__()
        self.linear1 = nn.Linear(15*15*128, 1024)
        self.activation1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 512)
        self.activation2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.drop1(self.activation1(x))
        x = self.linear2(x)
        x = self.drop2(self.activation2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.linear1 = nn.Linear(15*15*128, 1024)
        self.activation1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 512)
        self.activation2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.drop1(self.activation1(x))
        x = self.linear2(x)
        x = self.drop2(self.activation2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x

class ModelSubDep(nn.Module):

    def __init__(self):
        super(ModelSubDep, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size= 5, stride= 1, padding= 2)
        self.resblock1 = ResidualBlock(32,32,5)
        self.resblock2 = ResidualBlock(32,64,5)
        self.resblock3 = ResidualBlock(64,128,5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.labelclassifier = LabelClassifier()
        self.domainclassifier = DomainClassifier()

    def forward(self, x, alpha):
        x = self.conv1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.maxpool(x)
        F = nn.Flatten()
        x = F(x)
        y =  ReverseLayerF.apply(x, alpha)
        l = self.labelclassifier(x)
        d = self.domainclassifier(y)
        return l,d

