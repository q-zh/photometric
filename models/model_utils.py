import os
import torch
import torch.nn as nn

def getInput(args, data):
    input_list = [data['img']]
    if args.in_light: input_list.append(data['dirs'])
    if args.in_mask:  input_list.append(data['m'])
    return input_list

def parseData(args, sample, timer=None, split='train'):
    img, normal, mask = sample['img'], sample['normal'], sample['mask']
    ints = sample['ints']
    if args.in_light:
        dirs = sample['dirs'].expand_as(img)
    else: # predict lighting, prepare ground truth
        n, c, h, w = sample['dirs'].shape
        dirs_split = torch.split(sample['dirs'].view(n, c), 3, 1)
        dirs = torch.cat(dirs_split, 0)
    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
        dirs, ints = dirs.cuda(), ints.cuda()
        if timer: timer.updateTime('ToGPU')
    data = {'img': img, 'n': normal, 'm': mask, 'dirs': dirs, 'ints': ints}
    return data 

def parseData_stage2(args, sample, random_loc, split='train'):
    img_all = sample['img_all']
    if args.in_light:
        dirs = sample['dirs_all'].expand_as(img)
    else: # predict lighting, prepare ground truth
        n, c, h, w = sample['dirs_all'].shape
        dirs_split = torch.split(sample['dirs_all'].view(n, c), 3, 1)
    
    x_loc, y_loc = random_loc
    img_all_crop = img_all[:,:,x_loc - 8:x_loc + 8, y_loc - 8:y_loc + 8]
    del img_all
    if args.cuda:
        img_all_crop = img_all_crop.cuda()
    # img_all_crop = img_all_crop.repeat_interleave(32,2).repeat_interleave(32,3)
    n, c, h, w = img_all_crop.shape
    imgs = list(torch.split(img_all_crop, 3, 1))
    # ob_map_real = torch.zeros(n, 512, 512).cuda()
    for i in range(len(imgs)):
        img_patch = imgs[i].mean(1)
        img_patch = img_patch.repeat_interleave(32,1).repeat_interleave(32,2)
        dirs = dirs_split[i]
        if args.cuda:
            dirs = dirs.cuda()
        x= 0.5*(dirs[:,0]+1)*(32-1); 
        x=torch.round(x).type(torch.uint8).unsqueeze(1);
        x_one_hot = torch.zeros(n, 32).cuda().scatter_(1, x.long(), 1).unsqueeze(2).repeat(1,1,32)
        y= 0.5*(dirs[:,1]+1)*(32-1);
        y=torch.round(y).type(torch.uint8).unsqueeze(1);
        y_one_hot = torch.zeros(n, 32).cuda().scatter_(1, y.long(), 1).unsqueeze(1).repeat(1,32,1)
        loc_one_hot = x_one_hot * y_one_hot
        loc_one_hot = loc_one_hot.repeat(1,16,16)
        if i == 0:
            ob_map_real = img_patch * loc_one_hot
        else:
            ob_map_real,_ = torch.stack([ob_map_real, img_patch * loc_one_hot],1).max(1)
        # ob_map.append(img_patch * loc_one_hot)
        # for j in range(n):
        #     x= 0.5*(dirs[j,0]+1)*(32-1); 
        #     x=torch.round(x).type(torch.uint8);
        #     y= 0.5*(dirs[j,1]+1)*(32-1);
        #     y=torch.round(y).type(torch.uint8);
        #     for k in range(16):
        #         for l in range(16):
        #             ob_map_real[j,x + 32 * k, y + 32 * l] = img_patch[j,k,l]
    # ob_map_real = torch.stack(ob_map,1).sum(1)
    ob_map_real = ob_map_real.unsqueeze(1)
    # print (ob_map_real.shape)
    # print (ob_map_real.squeeze()[0,0:32,0:32])
    # if args.cuda:
    #     ob_map_real = ob_map_real.cuda() 
    return ob_map_real

def getInputChanel(args):
    args.log.printWrite('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        args.log.printWrite('[Network Input] Adding Light direction as input')
        c_in += 3
    if args.in_mask:
        args.log.printWrite('[Network Input] Adding Mask as input')
        c_in += 1
    args.log.printWrite('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records} # 'args': args}
    torch.save(state,   os.path.join(save_path, 'checkp_{}.pth.tar'.format(epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_{}_rec.pth.tar'.format(epoch)))

def conv_ReLU(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.ReLU(inplace=True)
                )

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = pad if pad >= 0 else (k - 1) // 2
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def outputConv(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True))

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )

def upconv(cin, cout):
    return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
