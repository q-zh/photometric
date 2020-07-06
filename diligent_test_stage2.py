import os
import torch
from models import model_utils
from utils import eval_utils, time_utils 
import numpy as np
from scipy import io

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, split, loader, models, log, epoch, recorder):
    models[0].eval()
    models[1].eval()
    log.printWrite('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    res = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            pred_c = models[0](input); timer.updateTime('Forward')
            input.append(pred_c)
            
            normals_gt = data['n']
            normals_fake = torch.zeros(1, 3, 128, 128)
            ob_map_sparse = torch.zeros(1, 1, 4096, 4096)
            ob_map_dense = torch.zeros(1, 1, 4096, 4096)
            ob_map_gt = torch.zeros(1, 1, 4096, 4096)
            if args.cuda:
                normals_fake = normals_fake.cuda()
                ob_map_sparse = ob_map_sparse.cuda()
                ob_map_dense = ob_map_dense.cuda()
                ob_map_gt = ob_map_gt.cuda()

            visited_times = torch.zeros(128,128)
            stride = 6
            # mask_ind_set = set(mask_ind)
            print('Object: ' + sample['obj'][0])
            # print(data['n'].shape)
            # print(data['img'].shape)
            with torch.no_grad():
                for x in range(0, 128 - 15, stride):
                    for y in range(0, 128 - 15, stride):
                        print ((x,y))
                        random_loc = torch.tensor([x + 8, y + 8])
                        input.append(random_loc)
                        pred = models[1](input);
                        input.pop();
                        normals_fake[0,:,x + 2:x + 14, y + 2:y + 14] += pred['n'][0,:, 2:14, 2:14]
                        visited_times[x + 2:x + 14, y + 2:y + 14] += 1
            timer.updateTime('Forward')
            normals_fake = normals_fake / 4

            # stride = 16
            # with torch.no_grad():
            #     for x in range(0, 128 - 15, stride):
            #         for y in range(0, 128 - 15, stride):
            #             print ((x,y))
            #             random_loc = torch.tensor([x + 8, y + 8])
            #             input.append(random_loc)
            #             # pred = models[1](input);
            #             pred = models[1](input);
            #             # ob_map_gt[:,:,x * 32: (x+16)*32,y*32:(y+16)*32] = model_utils.parseData_stage2(args, sample, random_loc, split)
            #             input.pop();
            #             ob_map_sparse[:,:,x * 32: (x+16)*32,y*32:(y+16)*32] += pred['ob_map_sparse']
            #             ob_map_dense[:,:,x * 32: (x+16)*32,y*32:(y+16)*32] += pred['ob_map_dense']
                        
            # width = 32;
            # dirs = data['dirs']
            # x= 0.5*(dirs[:,0]+1)*(width-1); 
            # x=torch.round(x).clamp(0, width - 1).long();
            # y= 0.5*(dirs[:,1]+1)*(width-1);
            # y=torch.round(y).clamp(0, width - 1).long();
            # idx_x = torch.split(x, 1, 0)
            # idx_y = torch.split(y, 1, 0)
            # stride = 16
            # print (x)
            # print (y)
            # if not os.path.exists(os.path.join(args.log_dir, 'test') + '/' + str(args.test_set) + '/' + sample['obj'][0]):
            #     os.mkdir(os.path.join(args.log_dir, 'test') + '/' + str(args.test_set) + '/' + sample['obj'][0])
            # d = os.path.join(args.log_dir, 'test') + '/' + str(args.test_set) + '/{}/mapind2.txt'
            # f = open(d.format(sample['obj'][0]), 'w')
            # dirs_x = torch.split(pred_c['dirs_x'], 1, 0)
            # dirs_y = torch.split(pred_c['dirs_y'], 1, 0)
            # for i in range(len(dirs_x)):
            #     _, x_idx = dirs_x[i].data.max(1)
            #     _, y_idx = dirs_y[i].data.max(1)
            #     f.write(str(x_idx.item() + 32 * y_idx.item() + 1) + '\n')
            # f.close()
            # with torch.no_grad():
            #     for x in range(0, 128 - 15, stride):
            #         for y in range(0, 128 - 15, stride):
            #             print ((x,y))
            #             random_loc = torch.tensor([x + 8, y + 8])
            #             input.append(random_loc)
            #             # pred = models[1](input);
            #             pred = models[1](input, idx_x, idx_y);
            #             input.pop();
            #             normals_fake[:,:,x:x + 16, y:y + 16] += pred['n']
            # #             # visited_times[x + 2:x + 14, y + 2:y + 14] += 1
            # timer.updateTime('Forward')
            # normals_fake = normals_fake / 4

            delta = angular_deviation(normals_fake, normals_gt)
            # normalfilepath =  os.path.join(args.log_dir, 'test') + '/' + str(args.test_set) + '/Images/' + sample['obj'][0] + '.mat'
            # normal_output = np.zeros([128, 128, 3])
            # normal_output[:, :, 0] = normals_fake[0, 0, :, :].cpu().numpy()
            # normal_output[:, :, 1] = normals_fake[0, 1, :, :].cpu().numpy()
            # normal_output[:, :, 2] = normals_fake[0, 2, :, :].cpu().numpy()
            # normal_gt = np.zeros([128, 128, 3])
            # normal_gt[:, :, 0] = normals_gt[0, 0, :, :].cpu().numpy()
            # normal_gt[:, :, 1] = normals_gt[0, 1, :, :].cpu().numpy()
            # normal_gt[:, :, 2] = normals_gt[0, 2, :, :].cpu().numpy()

            # ob_map_sparse_output = np.zeros([4096,4096])
            # ob_map_dense_output = np.zeros([4096,4096])
            # ob_map_sparse_output = ob_map_sparse[0,0,:,:].cpu().numpy()
            # ob_map_dense_output = ob_map_dense[0,0,:,:].cpu().numpy()
            # io.savemat(normalfilepath, {'normal_fake': normal_output, 'normal_gt': normal_gt, 'ob_map_sparse':ob_map_sparse_output[2048:2048+512,2048:2048+512],'ob_map_dense':ob_map_dense_output[2048:2048+512,2048:2048+512]})


            
            Dall = []
            for j in range(len(delta)):
                Dall.append(delta[j].unsqueeze(0))
            D_final = torch.cat(Dall,dim = 0 )
            d = os.path.join(args.log_dir, 'test') + '/' + str(args.test_set) + '/{}.txt'
            f = open(d.format(sample['obj'][0]), 'w')
            f.write(str(torch.mean(D_final)) + '\n')
            f.close()
    #         s2_est_obMp = True
    #         if s2_est_obMp:
    #             start_loc, end_loc = 32, 96
    #             random_loc = torch.randint(start_loc,end_loc,[2,1])
    #             input.append(random_loc)
    #             data['ob_map_real'] = model_utils.parseData_stage2(args, sample, random_loc, 'train')

    #         pred = models[1](input); timer.updateTime('Forward')
    #         input.pop()
    #         recoder, iter_res, error = prepareRes(args, data, pred_c, pred,random_loc, recorder, log, split)

    #         res.append(iter_res)
    #         iters = i + 1
    #         if iters % disp_intv == 0:
    #             opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
    #                     'timer':timer, 'recorder': recorder}
    #             log.printItersSummary(opt)

    #         if iters % save_intv == 0:
    #             results, nrow = prepareSave(args, data, pred_c, pred, random_loc)
    #             log.saveImgResults(results, split, epoch, iters, nrow=nrow, error='')
    #             log.plotCurves(recorder, split, epoch=epoch, intv=disp_intv)

    #         if stop_iters > 0 and iters >= stop_iters: break
    # res = np.vstack([np.array(res), np.array(res).mean(0)])
    # save_name = '%s_res.txt' % (args.suffix)
    # np.savetxt(os.path.join(args.log_dir, split, save_name), res, fmt='%.2f')
    # if res.ndim > 1:
    #     for i in range(res.shape[1]):
    #         save_name = '%s_%d_res.txt' % (args.suffix, i)
    #         np.savetxt(os.path.join(args.log_dir, split, save_name), res[:,i], fmt='%.3f')

    # opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    # log.printEpochSummary(opt)

def angular_deviation(iput, target):
    """Compute Regression loss."""
    X = []
    Y = []
    for i in range(iput.size(2)):
        for j in range(iput.size(3)):
            for k in range(iput.size(0)):
                if torch.norm(target[k,:,i,j]) > 0.1:
                    X.append(iput[k,:,i,j].unsqueeze(0))
                    Y.append(target[k,:,i,j].unsqueeze(0))

    x = torch.cat(X,dim=0)
    y = torch.cat(Y,dim=0)
    x_norm = torch.norm(x,dim=1)
    x_norm = x_norm.unsqueeze(1)
    x_norm = x_norm.expand(x_norm.size(0),x.size(1))
    y_norm = torch.norm(y,dim=1)
    y_norm = y_norm.unsqueeze(1)
    y_norm = y_norm.expand(y_norm.size(0),y.size(1))
    cosdelta = x*y/(x_norm*y_norm)
    cosdelta = torch.sum(cosdelta,dim=1)
    for ii in range(cosdelta.size(0)):
        if cosdelta[ii].data<=1 and cosdelta[ii].data>=-1:
            cosdelta[ii] = cosdelta[ii]
        else:
            cosdelta[ii] = torch.tensor(-1.).cuda()
    delta = torch.acos(cosdelta)
    avg_delta = sum(delta) / len(delta)
    return delta/np.pi*180

def prepareRes(args, data, pred_c, pred, random_loc, recorder, log, split):
    mask_var = data['m']
    data_batch = args.val_batch if split == 'val' else args.test_batch
    iter_res = []
    error = ''
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred_c['dirs'].data, data_batch)
        recorder.updateIter(split, l_acc.keys(), l_acc.values())
        iter_res.append(l_acc['l_err_mean'])
        error += 'D_%.3f-' % (l_acc['l_err_mean']) 

    if args.s2_est_n:
        random_x_loc, random_y_loc = random_loc
        n_tar = data['n'][:,:,random_x_loc - 8:random_x_loc + 8,random_y_loc - 8:random_y_loc + 8]
        mask_var = mask_var[:,:,random_x_loc - 8:random_x_loc + 8,random_y_loc - 8:random_y_loc + 8]
        acc, error_map = eval_utils.calNormalAcc(n_tar.data, pred['n'].data, mask_var.data)
        recorder.updateIter(split, acc.keys(), acc.values())
        iter_res.append(acc['n_err_mean'])
        error += 'N_%.3f-' % (acc['n_err_mean'])
        data['error_map'] = error_map['angular_map']
        
    return recorder, iter_res, error

def prepareSave(args, data, pred_c, pred, random_loc):
    # results = [data['img'].data, data['m'].data, (data['n'].data+1) / 2]
    input_var, mask_var = data['img'], data['m']
    results = [input_var.data, mask_var.data, (data['n'].data+1)/2]
    if args.s2_est_n:
        random_x_loc, random_y_loc = random_loc
        mask_var = mask_var[:,:,random_x_loc - 8:random_x_loc + 8,random_y_loc - 8:random_y_loc + 8]
        pred_n = (pred['n'].data + 1) / 2
        masked_pred = pred_n * mask_var.data.expand_as(pred['n'].data)
        res_n = [masked_pred, data['error_map']]
        results += res_n

    nrow = data['img'].shape[0]
    return results, nrow