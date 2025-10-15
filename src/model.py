import torch
import torch.nn as nn



#Conv Block
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        '''
        In channels is the number of input features channels 3 for rgb
        Out channels is the number of filters
        kernel size is the filter size
        stride is essentially how far the kernel move in each step, effectively similar to the lr
        activation is to whether apply activation or not
        '''
        
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn=nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.actfn = nn.SiLU(inplace=True) if activation else nn.Identity() #Activation Function
        
    def forward(self, x):
        return self.actfn(self.bn(self.conv(x)))



## Bottle Neck Block
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, skip=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride = 1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride = 1, padding=1)
        self.skip=skip
    def forward(self, x):
        x_ins = x #Residual connection
        x=self.conv1(x)
        x=self.conv2(x)
        if self.skip:
            x=x+x_ins # add residual connection if u want
        return x
### C2F