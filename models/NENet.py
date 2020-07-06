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

class FeatExtractor(nn.Module):
    def __init__(self, c_in = 4,conv_dim=16):
        
        super(FeatExtractor, self).__init__()

        layers = []
        layers.append(nn.Conv2d(c_in, conv_dim, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(3):
            if i in [0,1,2]:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
                layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
                layers.append(nn.ReLU(inplace=True))
                curr_dim = curr_dim * 2
            # else:
            #     layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1, bias=False))
            #     layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
            #     layers.append(nn.ReLU(inplace=True))
            #     curr_dim = curr_dim

        layers.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        output = self.main(x)
        return output

class FeatMerger(nn.Module):
    """Generator network."""
    def __init__(self, repeat_num=6):
        
        super(FeatMerger, self).__init__()

        curr_dim = 128
        layers = []
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(4):
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

# class FeatExtractor(nn.Module):
#     def __init__(self, batchNorm=False, c_in=3, other={}):
#         super(FeatExtractor, self).__init__()
#         self.other = other
#         self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
#         self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
#         self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
#         self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
#         self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
#         self.conv6 = model_utils.deconv(256, 128)
#         self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#         out = self.conv6(out)
#         out_feat = self.conv7(out)
#         n, c, h, w = out_feat.data.shape
#         out_feat   = out_feat.view(-1)
#         return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=32):
        super(Regressor, self).__init__()
               
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
#        
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

# class Regressor(nn.Module):
#     def __init__(self, batchNorm=False, other={}): 
#         super(Regressor, self).__init__()
#         self.other   = other
#         self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
#         self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
#         self.deconv3 = model_utils.deconv(128, 64)
#         self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
#         self.other   = other

#     def _make_output(self, cin, cout, k=3, stride=1, pad=1):
#         return nn.Sequential(
#                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

#     def forward(self, x, shape):
#         x      = x.view(shape[0], shape[1], shape[2], shape[3])
#         out    = self.deconv1(x)
#         out    = self.deconv2(out)
#         out    = self.deconv3(out)
#         normal = self.est_normal(out)
#         normal = torch.nn.functional.normalize(normal, 2, 1)
#         return normal

class NENet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=4, other={}):
        super(NENet, self).__init__()
        self.extractor = FeatExtractor(c_in)
        self.generator = FeatMerger()
        self.regressor = Regressor()
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        imgs = torch.split(x[0], 3, 1)
        idx = 1
        if self.other['in_light']: idx += 1
        if self.other['in_mask']:  idx += 1
        dirs_x = torch.split(x[idx]['dirs_x'], x[0].shape[0], 0)
        dirs_y = torch.split(x[idx]['dirs_y'], x[0].shape[0], 0)
        dirs = torch.split(x[idx]['dirs'], x[0].shape[0], 0)
        # ints = torch.split(x[idx]['intens'], 3, 1)
        random_x_loc, random_y_loc = x[idx + 1]
        # s2_inputs = []
        # for i in range(len(imgs)):
        #     n, c, h, w = imgs[i].shape
        #     l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
        #     # l_int = torch.diag(1.0 / (ints[i].contiguous().view(-1)+1e-8))
        #     # img   = imgs[i].contiguous().view(n * c, h * w)
        #     # img   = torch.mm(l_int, img).view(n, c, h, w)
        #     img_light = torch.cat([img, l_dir.expand_as(img)], 1)
        #     s2_inputs.append(img_light)
        # return s2_inputs
        s2_inputs = []
        tmp = []
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            dirs_map = nn.functional.softmax(dirs_x[i],1).unsqueeze(2).repeat(1,1,dirs_x[i].shape[1]) * nn.functional.softmax(dirs_y[i],1).unsqueeze(1).repeat(1,dirs_y[i].shape[1],1)
            dirs_map = dirs_map.repeat(1,16,16).unsqueeze(1)
            dirs_map = dirs_map.cuda()
            # l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
            # l_int = torch.diag(1.0 / (ints[i].contiguous().view(-1)+1e-8))
            # img   = imgs[i].contiguous().view(n * c, h * w)
            # img   = torch.mm(l_int, img).view(n, c, h, w)
            img = imgs[i][:,:,random_x_loc - 8:random_x_loc + 8,random_y_loc - 8:random_y_loc + 8]
            img = img.repeat_interleave(32,2).repeat_interleave(32,3)
            # img = img.mean(1)
            # img = img.unsqueeze(1)
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
        feats = torch.Tensor()
        for i in range(len(inputs)):
            feat = self.extractor(inputs[i])
            if i == 0:
                feats = feat
            else:
                if self.fuse_type == 'mean':
                    feats = torch.stack([feats, feat], 1).sum(1)
                elif self.fuse_type == 'max':
                    feats, _ = torch.stack([feats, feat], 1).max(1)
        if self.fuse_type == 'mean':
            feats = feats / len(img_split)
        feat_fused = feats
        ob_map_dense = self.generator(feat_fused)
        # print (ob_map_dense.shape)
        # print (ob_map_parse.shape)
        ob_map = torch.cat([ob_map_parse,ob_map_dense], 1)
        normal = self.regressor(ob_map)
        pred = {}
        pred['ob_map_dense'] = ob_map_dense
        pred['n'] = normal
        return pred
