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



### SPPF Block

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__() #kernel_Size=maxpool_size
        hidden_channels = in_channels//2 #
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1,padding=0)
        self.conv2 = Conv(4*hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.mp = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, dilation=1, ceil_mode=False) #padding is set to kernelsize//2 to keep the output dimension the same.
    def forward(self, x):
        x = self.conv1(x)
        #apply maxpool for 3 scales
        y = x
        outputs = [x]
        for i in range(3):
            y = self.mp(y)
            outputs.append(y)
        x = torch.cat(outputs, dim=1)
        x = self.conv2(x)
        return x
#testing testing
sppf = SPPF(in_channels=64, out_channels=128)
x = torch.rand(1, 64, 64, 64)
y = sppf(x)
print("hello??")
print(f"output shape: {y.shape}")





#Backbone

def yolo_parameters(version):
    #return d, w, r based on version
    match version:
        case "n":
            return 1/3,1/4,2.0
        case "s":
            return 1/3,1/2,2.0
        case "m":
            return 2/3,3/4,1.5
        case "l":
            return 1.0,1.25,1.0
        case "x":
            return 1.0,1.25,1.0  
        case _:
            print(f"Unknown YOLO Version: {version} falling back to NANO")
            return 1/3,1/4,2.0
class Backbone(nn.Module):
    def __init__(self, version, in_channels=3, skip_bnck = True):
        super().__init__()
        d, w, r = yolo_parameters(version)
        
        ###Conv Layers
        self.conv_0 = Conv(in_channels, int(64*w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128*w), int(256*w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1)
        
        
        #### C2F Layers
        self.c2f_2 = C2F(int(128*w), int(128*w), n_bnck=int(3*d),skip=True)
        self.c2f_4 = C2F(int(256*w), int(256*w), n_bnck=int(6*d),skip=True)
        self.c2f_6 = C2F(int(512*w), int(512*w), n_bnck=int(6*d),skip=True)
        self.c2f_8 = C2F(int(512*w*r), int(512*w*r), n_bnck=int(3*d),skip=True)
        ### SPPF layer
        self.sppf_9 = SPPF(int(512*w*r), int(512*w*r))
    def forward(self, x):
        x=self.conv0(x)
        x=self.conv1(x)
        x=self.c2f_2(x)
        x=self.conv_3(x)
        x1 = self.c2f_4(x) #keep for some other thing i forgot
        x=self.conv_5(x1)
        x2=self.c2f_6(x) #Also keep it for some concatenation, still didnt reach it
        x=self.conv_7(x2)
        x=self.c2f_8(x)
        x3=self.sppf_9(x) # same thing
        
        return x1,x2,x3


