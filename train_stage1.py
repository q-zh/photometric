from models import model_utils
from utils  import eval_utils, time_utils

def train(args, loader, model, criterion, optimizer, log, epoch, recorder):
    model.train()
    log.printWrite('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        data = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)
        pred = model(input); timer.updateTime('Forward')

        optimizer.zero_grad()
        loss = criterion.forward(pred, data); 
        timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')

        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)

        if iters % args.train_save == 0:
            results, recorder, nrow = prepareSave(args, data, pred, recorder, log) 
            log.saveImgResults(results, 'train', epoch, iters, nrow=nrow)
            log.plotCurves(recorder, 'train', epoch=epoch, intv=args.train_disp)

        if args.max_train_iter > 0 and iters >= args.max_train_iter: break
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(args, data, pred, recorder, log):
    results = [data['img'].data, data['m'].data, (data['n'].data+1)/2]
    if args.s1_est_d:
        l_acc, data['dir_err'] = eval_utils.calDirsAcc(data['dirs'].data, pred['dirs'].data, args.batch)
        recorder.updateIter('train', l_acc.keys(), l_acc.values())

    nrow = data['img'].shape[0] if data['img'].shape[0] <= 32 else 32
    return results, recorder, nrow
