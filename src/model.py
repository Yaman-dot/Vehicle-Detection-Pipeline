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
### C2F = conv+bottlenec*n+conv

class C2F(nn.Module):
    def __init__(self, in_channels, out_channels, n_bnck=1, skip=True):
        super().__init__()
        self.mid_channels = out_channels//2
        self.n_bnck = n_bnck
        
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        #Bottleneck layers
        self.bnck_layers = nn.ModuleList([BottleNeck(self.mid_channels, self.mid_channels) for _ in range(n_bnck)])
        self.conv2 = Conv((n_bnck+2)*self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x=self.conv1(x)
        #split
        '''mid = x.shape[1]//2
        x1,x2 = x[:,:mid,:,:], x[:,mid,:,:]'''
        try:
            x1,x2 = torch.chunk(x,2,dim=1)
        except RuntimeError:
            print("It probably can't divide it by 2 evenly, using manual slicing instead")
            mid = x.shape[1]//2
            x1,x2 = x[:, :mid, :, :], x[:,mid:,:,:]
    
        outputs = [x1,x2]
        for bnck in self.bnck_layers:
            x2 = bnck(x2)
            outputs.append(x2)
        
        x = torch.cat(outputs, dim=1)
        x = self.conv2(x)
        return x

c2f = C2F(in_channels=64, out_channels=128, n_bnck=2)

x = torch.rand(1, 64, 244, 244)
y = c2f(x)
print("hello??")
print(f"output shape: {y.shape}")