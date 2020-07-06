import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

# class Generator(nn.Module):
#     def __init__(self,conv_dim = 16, repeat_num = 3):
#         super(Generator, self).__init__()

#         layers1 = []
#         layers1.append(nn.Conv2d(1, conv_dim, kernel_size = 7, stride = 1, padding = 3, bias = False))
#         layers1.append(nn.InstanceNorm2d(conv_dim, affine = True, track_running_stats = True))
#         layers1.append(nn.ReLU(inplace=True))

#         curr_dim = conv_dim
#         for i in range(3):
#             layers1.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size = 4, stride = 2, padding = 1, bias = False))
#             layers1.append(nn.InstanceNorm2d(curr_dim * 2, affine = True, track_running_stats = True))
#             layers1.append(nn.ReLU(inplace = True))
#             curr_dim = curr_dim * 2

#         layers2 = []
#         for i in range(3):
#             layers2.append(nn.Conv2d(curr_dim, curr_dim *2, kernel_size = 4, stride = 2, padding = 1, bias = False))
#             layers2.append(nn.InstanceNorm2d(curr_dim *2, affine = True, track_running_stats = True))
#             layers2.append(nn.ReLU(inplace = True))
#             curr_dim = curr_dim * 2

#         for i in range(3):
#             layers2.append(ResidualBlock(dim_in = curr_dim, dim_out = curr_dim))

#         for i in range(3):
#             layers2.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size = 4, stride =2,padding = 1, bias = False))
#             layers2.append(nn.InstanceNorm2d(curr_dim//2,affine = True, track_running_stats = True))
#             layers2.append(nn.ReLU(inplace = True))
#             curr_dim = curr_dim // 2

#         layers3 = []
#         curr_dim = curr_dim * 2
#         for i in range(repeat_num):
#             layers3.append(ResidualBlock(dim_in = curr_dim, dim_out = curr_dim))

#         for i in range(3):
#             layers3.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size = 4, stride = 2, padding = 1, bias = False))
#             layers3.append(nn.InstanceNorm2d(curr_dim//2,affine=True,track_running_stats = True))
#             layers3.append(nn.ReLU(inplace = True))
#             curr_dim = curr_dim//2

#         layers3.append(nn.Conv2d(curr_dim, 1, kernel_size = 7, stride = 1, padding = 3, bias = False))
#         layers3.append(nn.ReLU(inplace = True))
#         self.main1 = nn.Sequential(*layers1)
#         self.main2 = nn.Sequential(*layers2)
#         self.main3 = nn.Sequential(*layers3)

#         mv = torch.arange(1024)
#         X = mv.float()%32
#         Y = torch.floor(mv.float()/32)
#         x = X/31*2 - 1
#         y = Y/32*2 - 1
#         x = x.unsqueeze(1)
#         y = y.unsqueeze(1)
#         r = torch.cat([x,y],dim = 1)
#         r_norm = torch.norm(r,dim = 1)
#         mask = torch.lt(r_norm,1)
#         self.mask = torch.reshape(mask,[32,32])
#         self.mask = self.mask.repeat(16,16)


#     def forward(self,x):
#         mask = self.mask
#         mask = mask.expand(x.size(0),1,mask.size(0),mask.size(1))
#         x1 = torch.split(x, 32, dim = 2)
#         s1 = []
#         for t1 in x1:
#             x2 = torch.split(t1 ,32, dim = 3)
#             s2 = []
#             for t2 in x2:
#                 y = self.main1(t2)
#                 s2.append(y)
#             y2 = torch.cat(s2, dim = 3)
#             s1.append(y2)
#         y1 = torch.cat(s1, dim = 2)
#         z1 = self.main2(y1)
#         u1 = torch.cat([y1,z1], dim = 1)
#         v1 = self.main3(u1)
#         w = torch.zeros(v1.size()).cuda()
#         w[mask] = v1[mask]
#         return w

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=32, repeat_num=6):
        
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)
        
        mv = torch.arange(1024)
        X = mv.float()%32
        Y = torch.floor(mv.float()/32)
        x = X/31*2-1
        y = Y/31*2-1
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        r = torch.cat([x,y],dim=1)
        r_norm = torch.norm(r,dim=1)
        mask = torch.lt(r_norm,1)
        self.mask = torch.reshape(mask,[32,32])
        self.mask = self.mask.repeat(16, 16)
        


    def forward(self, x):
        mask = self.mask
        mask = mask.expand(x.size(0),1,mask.size(0),mask.size(1))
        x1 = self.main(x)
        y = torch.zeros(x1.size()).cuda()
        y[mask] = x1[mask]
        return y

# class Discriminator(nn.Module):
#     """Discriminator network with PatchGAN."""
#     def __init__(self, image_size=32):
#         super(Discriminator, self).__init__()
               
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        
#         block1_1 = []
#         block1_1.append(nn.ReLU())
#         block1_1.append(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
#         block1_1.append(nn.Dropout2d(p=0.2))
#         self.denseblock1_1 = nn.Sequential(*block1_1)
        
#         block1_2 = []
#         block1_2.append(nn.ReLU())
#         block1_2.append(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1))
#         block1_2.append(nn.Dropout2d(p=0.2))
#         self.denseblock1_2 = nn.Sequential(*block1_2)                
        
#         trans = []
#         trans.append(nn.ReLU())
#         trans.append(nn.Conv2d(48, 48, kernel_size=1, stride=1))
#         trans.append(nn.Dropout2d(p=0.2))
#         trans.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
#         self.transition = nn.Sequential(*trans)
        
#         block2_1 = []
#         block2_1.append(nn.ReLU())
#         block2_1.append(nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1))
#         block2_1.append(nn.Dropout2d(p=0.2))
#         self.denseblock2_1 = nn.Sequential(*block2_1)
        
#         block2_2 = []
#         block2_2.append(nn.ReLU())
#         block2_2.append(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1))
#         block2_2.append(nn.Dropout2d(p=0.2))
#         self.denseblock2_2 = nn.Sequential(*block2_2)    

#         self.conv2 = nn.Conv2d(80,80,kernel_size = 1, stride = 1,padding = 0)
#         dense = []
#         dense.append(nn.Linear(20480,32)) 
#         self.dense = nn.Sequential(*dense)
        
#         layers = []
#         curr_dim = 32
#         for i in range(2):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
#             layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim * 2   
#         for i in range(3):
#             layers.append(ResidualBlock(dim_in = curr_dim, dim_out = curr_dim))
#         for i in range(2):
#             layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size = 4, stride = 2, padding= 1, bias = False))
#             layers.append(nn.InstanceNorm2d(curr_dim//2, affine = True, track_running_stats=True))
#             layers.append(nn.ReLU(inplace=True))
#             curr_dim = curr_dim//2
#         self.layers = nn.Sequential(*layers)

#         self.conv3 = nn.Conv2d(curr_dim * 2, 3, kernel_size=1, stride=1, padding=0)
# #        
#     def forward(self, x):
#         x1 = torch.split(x,32,dim = 2)
#         s1 = []
#         for t1 in x1:
#             x2 = torch.split(t1, 32, dim = 3)
#             s2 = []
#             for t2 in x2:
#                 h1 = self.conv1(t2)
#                 h2 = torch.cat([h1, self.denseblock1_1(h1)], dim=1)
#                 h3 = torch.cat([h1, self.denseblock1_1(h1), self.denseblock1_2(h2)], dim=1)
#                 d1 = self.transition(h3)
#                 d2 = torch.cat([d1, self.denseblock2_1(d1)], dim=1)
#                 d3 = torch.cat([d1, self.denseblock2_1(d1), self.denseblock2_2(d2)], dim=1)
#                 d4 = self.conv2(d3)
#                 d = d4.view(-1,20480)
#                 s2.append(self.dense(d).unsqueeze(2).unsqueeze(3))
#             y2 = torch.cat(s2, dim = 3)
#             s1.append(y2)
#         y1 = torch.cat(s1, dim = 2)
#         z1 = self.layers(y1)
#         u1 = torch.cat([y1,z1],dim = 1)
#         v1 = self.conv3(u1)
#         out_reg = torch.nn.functional.normalize(v1, 2, 1)
#         return out_reg

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=32):
        super(Discriminator, self).__init__()
               
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        
        block1_1 = []
        block1_1.append(nn.ReLU())
        block1_1.append(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))
        block1_1.append(nn.Dropout2d(p=0.2))
        self.denseblock1_1 = nn.Sequential(*block1_1)
        
        block1_2 = []
        block1_2.append(nn.ReLU())
        block1_2.append(nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1))
        block1_2.append(nn.Dropout2d(p=0.2))
        self.denseblock1_2 = nn.Sequential(*block1_2)                
        
        trans = []
        trans.append(nn.ReLU())
        trans.append(nn.Conv2d(48, 48, kernel_size=1, stride=1))
        trans.append(nn.Dropout2d(p=0.2))
        trans.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
        self.transition = nn.Sequential(*trans)
        
        block2_1 = []
        block2_1.append(nn.ReLU())
        block2_1.append(nn.Conv2d(48, 16, kernel_size=3, stride=1, padding=1))
        block2_1.append(nn.Dropout2d(p=0.2))
        self.denseblock2_1 = nn.Sequential(*block2_1)
        
        block2_2 = []
        block2_2.append(nn.ReLU())
        block2_2.append(nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1))
        block2_2.append(nn.Dropout2d(p=0.2))
        self.denseblock2_2 = nn.Sequential(*block2_2)        
        
        layers = []
        curr_dim = 80
        for i in range(3):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2   
        self.layers = nn.Sequential(*layers)
        
        self.pooling = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(curr_dim, 3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        h1 = self.conv1(x)
        h2 = torch.cat([h1, self.denseblock1_1(h1)], dim=1)
        h3 = torch.cat([h1, self.denseblock1_1(h1), self.denseblock1_2(h2)], dim=1)
        d1 = self.transition(h3)
        d2 = torch.cat([d1, self.denseblock2_1(d1)], dim=1)
        d3 = torch.cat([d1, self.denseblock2_1(d1), self.denseblock2_2(d2)], dim=1)        
        d4 = self.layers(d3)
        d4 = self.pooling(d4)
        out_reg = self.conv2(d4)
        out_reg = torch.nn.functional.normalize(out_reg, 2, 1)
        return out_reg

class NENet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=4, other={}):
        super(NENet, self).__init__()
        self.generator = Generator()
        self.regressor = Discriminator()
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other


    def prepareInputs(self, x):
        imgs = torch.split(x[0], 3, 1)
        idx = 1
        if self.other['in_light']: idx += 1
        if self.other['in_mask']:  idx += 1
        dirs_x = torch.split(x[idx]['dirs_x'], x[0].shape[0], 0)
        dirs_y = torch.split(x[idx]['dirs_y'], x[0].shape[0], 0)
        dirs = torch.split(x[idx]['dirs'], x[0].shape[0], 0)
        random_x_loc, random_y_loc = x[idx + 1]
        s2_inputs = []
        tmp = []
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            dirs_map = nn.functional.softmax(dirs_x[i],1).unsqueeze(2).repeat(1,1,dirs_x[i].shape[1]) * nn.functional.softmax(dirs_y[i],1).unsqueeze(1).repeat(1,dirs_y[i].shape[1],1)
            dirs_map = dirs_map.repeat(1,16,16).unsqueeze(1)
            dirs_map = dirs_map.cuda()
            img = imgs[i][:,:,random_x_loc - 8:random_x_loc + 8,random_y_loc - 8:random_y_loc + 8]
            img = img.repeat_interleave(32,2).repeat_interleave(32,3)
            img_light = torch.cat([img, dirs_map], 1)
            s2_inputs.append(img_light)

            _, x_idx = dirs_x[i].data.max(1)
            _, y_idx = dirs_y[i].data.max(1)
            x=x_idx.type(torch.uint8).unsqueeze(1);
            x_one_hot = torch.zeros(n, 32).cuda().scatter_(1, x.long(), 1).unsqueeze(2).repeat(1,1,32)
            y=y_idx.type(torch.uint8).unsqueeze(1);
            y_one_hot = torch.zeros(n, 32).cuda().scatter_(1, y.long(), 1).unsqueeze(1).repeat(1,32,1)
            loc_one_hot = x_one_hot * y_one_hot
            max_filter = loc_one_hot.repeat(1,16,16)
            max_filter = max_filter.cuda()
            img_gray = img.mean(1)
            img_gray_filtered = img_gray * max_filter
            tmp.append(img_gray_filtered)
        regressor_inputs,_ = torch.stack(tmp,1).max(1)
        regressor_inputs = regressor_inputs.unsqueeze(1)
        return s2_inputs, regressor_inputs

    def forward(self, x):
        inputs, ob_map_parse = self.prepareInputs(x)
        ob_map_dense = self.generator(ob_map_parse)
        ob_map = torch.cat([ob_map_parse,ob_map_dense], 1)
        normal = self.regressor(ob_map)
        pred = {}
        pred['ob_map_dense'] = ob_map_dense
        pred['n'] = normal
        pred['ob_map_sparse'] = ob_map_parse
        return pred