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
'''#testing testing
sppf = SPPF(in_channels=64, out_channels=128)
x = torch.rand(1, 64, 64, 64)
y = sppf(x)
print("hello??")
print(f"output shape: {y.shape}")'''





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
        x=self.conv_0(x)
        x=self.conv_1(x)
        x=self.c2f_2(x)
        x=self.conv_3(x)
        x1 = self.c2f_4(x) #keep for some other thing i forgot
        x=self.conv_5(x1)
        x2=self.c2f_6(x) #Also keep it for some concatenation, still didnt reach it
        x=self.conv_7(x2)
        x=self.c2f_8(x)
        x3=self.sppf_9(x) # same thing
        
        return x1,x2,x3

#testing
'''Backbone_n = Backbone(version='n')


x =torch.rand(1,3,640,640)
x1,x2,x3 = Backbone_n(x)
print(f"{x1.shape}\n,{x2.shape}\n,{x3.shape}")'''


###NECK BUT IT NEEDS UPSAMPLE +C2F 
#####UPSAMPLE
class Upsample(nn.Module):
    def __init__(self, scale_factor=2,mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)




######NECK
class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d,w,r = yolo_parameters(version=version)
        ###Upsampling Layerss
        self.upsample_10_13 = Upsample()#No trainable params
        
        ## C2F Layers
        self.c2f_12 = C2F(in_channels=int(512*w*(1+r)), out_channels=int(512*w), 
                            n_bnck=int(3*d), skip=False)
        self.c2f_15 = C2F(in_channels=int(768*w), out_channels=int(256*w),
                            n_bnck=int(3*d), skip=False)
        self.c2f_18 = C2F(in_channels=int(768*w), out_channels=int(512*w),
                            n_bnck=int(3*d), skip=False)
        self.c2f_21 = C2F(in_channels=int(512*w*(1+r)), out_channels=int(512*w*r),
                            n_bnck=int(3*d), skip=False)
        #CONV Layers
        self.conv_16 = Conv(in_channels=int(256*w), out_channels=int(256*w), kernel_size=3, stride=2, padding=1)
        self.conv_19 = Conv(in_channels=int(512*w), out_channels=int(512*w), kernel_size=3, stride=2, padding=1)
    def forward(self, x_res1, x_res2, x):
        res1 = x
        
        x = self.upsample_10_13(x)
        x = torch.cat([x, x_res2], dim=1)
        
        res2 = self.c2f_12(x) #residue connection cuh
        
        x = self.upsample_10_13(res2)
        x = torch.cat([x, x_res1], dim=1)
        
        out_1 = self.c2f_15(x)
        
        x=self.conv_16(out_1)

        x = torch.cat([x, res2], dim=1)
        out_2 = self.c2f_18(x)
        
        x = self.conv_19(out_2)
        
        x = torch.cat([x, res1], dim=1)
        out_3 = self.c2f_21(x)
        
        return out_1, out_2, out_3


####TEEEEEEEST
neck = Neck(version="n")

x = torch.rand(1, 3, 640, 640)
out1, out2, out3 = Backbone(version='n')(x)
out1,out2,out3 = neck(out1, out2, out3)
#print(out1.shape)
#print(out2.shape)
#print(out3.shape)



####Head, need to build the dfl first
######DFL
class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch

        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)

        x=torch.arange(ch, dtype=torch.float).view(1,ch,1,1)
        self.conv.weight.data[:]=torch.nn.Parameter(x)

    def forward(self, x):
        b,c,a=x.shape #x have 4*ch num of channels: x=[bs,4*ch,c]
        x=x.view(b,4,self.ch,a).transpose(1,2) #[bs,ch,4,a]

        #take the softmax on channel dimension to get distribution probabilities
        x=x.softmax(1) 
        x=self.conv(x) #[b, 1,4,a]
        return x.view(b,4,a) #[b,4,a]
#testing
'''dfl = DFL()
x = torch.rand(1,64,128)
y = dfl(x)
print(dfl)'''


###Head
class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes = 80):
        super().__init__()
        
        self.ch = ch
        self.coordinates = self.ch*4 #number of bounding box coordinates
        self.nc = num_classes
        self.no = self.coordinates+self.nc #number of output per anchor box
        
        self.stride = torch.zeros(3) #calculated during training
        d, w, r = yolo_parameters(version=version)
        
        ##for bounding boxes
        self.box = nn.ModuleList([
            nn.Sequential(Conv(int(256*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                          Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            
            nn.Sequential(Conv(int(512*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                          Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            
            nn.Sequential(Conv(int(512*w*r), self.coordinates, kernel_size=3, stride=1, padding=1),
                          Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
        ])
        
        
        ##for classes
        self.cls = nn.ModuleList([
            nn.Sequential(Conv(int(256*w), self.nc, kernel_size=3, stride=1, padding=1),
                          Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(Conv(int(512*w), self.nc, kernel_size=3, stride=1, padding=1),
                          Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(Conv(int(512*w*r), self.nc, kernel_size=3, stride=1, padding=1),
                          Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                          nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
        ])
        
        self.dfl = DFL()
    def forward(self, x):
        #x is output of the neck which consists of 3 tensors with different res and channel dim
        for i in range(len(self.box)):
            box=self.box[i](x[i]) #[bs, num_coordinates, w, h]
            cls=self.cls[i](x[i]) #[bs, num_classes, w, h]
            x[i]=torch.cat((box,cls),dim=1) #[bs, num_coordinates+num_classes,w,h]
        #No DFL output when training
        if self.training:
            return x #[3,bs,num_coordinates+num_classes,w,h]
        
        anchors, strides = (i.transpose(0,1) for i in self.make_anchors(x,self.stride))
        
        x=torch.cat([i.view(x[0].shape[0], self.no,-1) for i in x], dim=2) #[bs, 4*self.ch+self.nc, sum_i(h[i],w[i])]
        
        box,cls = x.split(split_size=(4*self.ch, self.nc), dim=1)
        a,b=self.dfl(box).chunk(2,1) #a=b=[bs,2*self.ch,sum_i(h[i],w[i])]
        a=anchors.unsqueeze(0)-a
        b=anchors.unsqueeze(0)+b
        box=torch.cat(tensors=((a+b)/2, b-1), dim=1)
        
        return torch.cat(tensors=(box*strides, cls.sigmoid()), dim=1)
    def make_anchors(x, stride, offset=0.5):
        assert x is not None
        anchor_Tensor, stride_Tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(stride):
            _,_,h,w = x[i].shape
            sx = torch.arange(end=w,device=device, dtype=dtype)
            sy = torch.arange(end=h, device=device, dtype=dtype)
            anchor_Tensor.append(torch.stack((sx,sy),-1).view(-1,2))
            stride_Tensor.append(torch.full((h*w,1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_Tensor), torch.cat(stride_Tensor)
    
    
##Test
'''head = Head(version='n')
y = head([out1, out2, out3])
print(out1.shape)
print(out2.shape)
print(out3.shape)'''

##YOLO

class Yolo(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version)
    def forward(self, x):
        x=self.backbone(x)
        x = self.neck(x[0],x[1],x[2])
        return self.head(list(x))