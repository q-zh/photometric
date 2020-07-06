import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
from utils import eval_utils

class SoftmaxProbability2D(nn.Module):
    def __init__(self):
        super(SoftmaxProbability2D, self).__init__()

    def forward(self, x):
        origin_shape = x.data.shape
        seq_x = []
        for channel_ix in range(origin_shape[1]):
            softmax_ = nn.functional.softmax(x[:, channel_ix, :, :].contiguous().view((origin_shape[0],origin_shape[2]*origin_shape[3])),dim = 1)\
            .view((origin_shape[0], origin_shape[2], origin_shape[3]))
            seq_x.append(softmax_)
        x = torch.stack(seq_x, dim=1)
        return x

# Classification
class FeatExtractor(nn.Module):
    def __init__(self, batchNorm, c_in, c_out=256):
        super(FeatExtractor, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,    k=3, stride=2, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128,   k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 128,  128,   k=3, stride=1, pad=1)
        self.conv6 = model_utils.conv(batchNorm, 128,  256,   k=3, stride=2, pad=1)
        self.conv7 = model_utils.conv(batchNorm, 256,  256,   k=3, stride=1, pad=1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        return out

class Classifier(nn.Module):
    def __init__(self, batchNorm, c_in, other):
        super(Classifier, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, 512,  256, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)
        self.other = other
        
        self.dir_x_est= nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, 32, k=1, stride=1, pad=0))

        self.dir_y_est= nn.Sequential(
                    model_utils.conv(batchNorm, 256, 64,  k=1, stride=1, pad=0),
                    model_utils.outputConv(64, 32, k=1, stride=1, pad=0))


    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        outputs = {}
        if self.other['s1_est_d']:
            # outputs['dir_x'] = nn.functional.softmax(self.dir_x_est(out).squeeze(),dim =1)
            # outputs['dir_y'] = nn.functional.softmax(self.dir_y_est(out).squeeze(),dim =1)
            outputs['dir_x'] = self.dir_x_est(out).squeeze(2).squeeze(2)
            outputs['dir_y'] = self.dir_y_est(out).squeeze(2).squeeze(2)
            
        return outputs

class LCNet(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(LCNet, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in, 128)
        self.classifier = Classifier(batchNorm, 256, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other     = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        n, c, h, w = x[0].shape
        t_h, t_w = self.other['test_h'], self.other['test_w']
        if (h == t_h and w == t_w):
            imgs = x[0] 
        else:
            print('Rescaling images: from %dX%d to %dX%d' % (h, w, t_h, t_w))
            imgs = torch.nn.functional.upsample(x[0], size=(t_h, t_w), mode='bilinear')

        inputs = list(torch.split(imgs, 3, 1))
        idx = 1
        if self.other['in_light']:
            light = torch.split(x[idx], 3, 1)
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], light[i]], 1)
            idx += 1
        if self.other['in_mask']:
            mask = x[idx]
            if mask.shape[2] != inputs[0].shape[2] or mask.shape[3] != inputs[0].shape[3]:
                mask = torch.nn.functional.upsample(mask, size=(t_h, t_w), mode='bilinear')
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], mask], 1)
            idx += 1
        return inputs

    def fuseFeatures(self, feats, fuse_type):
        if fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        return feat_fused

    def convertMidDirs(self, pred):
        _, x_idx = pred['dirs_x'].data.max(1)
        _, y_idx = pred['dirs_y'].data.max(1)
        dirs = eval_utils.SphericalClassToDirs(x_idx, y_idx, self.other['dirs_cls'])
        return dirs

    def convertDirs(self, pred):
        _, x_idx = pred['dirs_x'].data.max(1)
        _, y_idx = pred['dirs_y'].data.max(1)
        dirs = eval_utils.SphericalLocToDirs(x_idx, y_idx, 32)
        return dirs

    def convertMidIntens(self, pred, img_num):
        _, idx = pred['ints'].data.max(1)
        ints = eval_utils.ClassToLightInts(idx, self.other['ints_cls'])
        ints = ints.view(-1, 1).repeat(1, 3)
        ints = torch.cat(torch.split(ints, ints.shape[0] // img_num, 0), 1)
        return ints

    def forward(self, x):
        inputs = self.prepareInputs(x)
        feats = []
        for i in range(len(inputs)):
            out_feat = self.featExtractor(inputs[i])
            shape    = out_feat.data.shape
            feats.append(out_feat)
        feat_fused = self.fuseFeatures(feats, self.fuse_type)

        l_dirs_x, l_dirs_y, l_dirs_map = [], [],[]
        for i in range(len(inputs)):
            net_input = torch.cat([feats[i], feat_fused], 1)
            outputs = self.classifier(net_input)
            if self.other['s1_est_d']:
                l_dirs_x.append(outputs['dir_x'])
                l_dirs_y.append(outputs['dir_y'])
                # outputs_shape = outputs['dir_x'].shape
                # map = torch.mul(outputs['dir_x'].unsqueeze(2).repeat(1,1,outputs_shape[1]),outputs['dir_y'].unsqueeze(1).repeat(1,outputs_shape[1]),1) 
                # l_dirs_map.append(map)

        pred = {}
        if self.other['s1_est_d']:
            pred['dirs_x'] = torch.cat(l_dirs_x, 0)
            pred['dirs_y'] = torch.cat(l_dirs_y, 0)
            pred['dirs'] = self.convertDirs(pred)
            # pred['dirs_map'] = torch.cat(map,0)
        return pred
