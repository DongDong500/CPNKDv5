import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
from metrics import StreamSegMetrics
from _get_dataset import _get_dataset
from _loop_eval import _validate
from _loop_train import _accumulate
from _load_model import _load_model

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def _train(opts, devices, run_id) -> dict:

    logdir = os.path.join(opts.Tlog_dir, 'run_' + str(run_id).zfill(2))
    writer = SummaryWriter(log_dir=logdir) 

    ### (1) Get datasets

    strain_dst, sval_dst, stest_dst = _get_dataset(opts, opts.s_dataset, opts.s_dataset_ver)
    
    s_train_loader = DataLoader(strain_dst, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    s_val_loader = DataLoader(sval_dst, batch_size=opts.val_batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    s_test_loader = DataLoader(stest_dst, batch_size=opts.test_batch_size, num_workers=opts.num_workers, 
                                shuffle=True, drop_last=True)

    ttrain_dst, tval_dst, ttest_dst = _get_dataset(opts, opts.t_dataset, opts.t_dataset_ver)
    
    t_train_loader = DataLoader(ttrain_dst, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    t_val_loader = DataLoader(tval_dst, batch_size=opts.val_batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    t_test_loader = DataLoader(ttest_dst, batch_size=opts.test_batch_size, num_workers=opts.num_workers, 
                                shuffle=True, drop_last=True)

    ### (2) Set up criterion

    if opts.loss_type == 'kd_loss':
        criterion = utils.KDLoss(alpha=opts.alpha, temperature=opts.T)
    elif opts.loss_type == 'gp_loss':
        criterion = utils.GPLoss()
    else:
        raise NotImplementedError

    ### (3 -1) Load teacher & student models

    s_model = _load_model(opts=opts, model_name=opts.s_model, verbose=True,
                            msg=" Primary model selection: {}".format(opts.s_model),
                            output_stride=opts.output_stride, sep_conv=opts.separable_conv)
    t_model = _load_model(opts=opts, model_name=opts.t_model, verbose=True,
                            msg=" Auxiliary model selection: {}".format(opts.t_model),
                            output_stride=opts.t_output_stride, sep_conv=opts.t_separable_conv)

    ### (4) Set up optimizer

    if opts.s_model.startswith("deeplab"):
        if opts.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': s_model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': s_model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError
    else:
        optimizer = optim.RMSprop(s_model.parameters(), 
                                    lr=opts.lr, 
                                    weight_decay=opts.weight_decay,
                                    momentum=opts.momentum)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                step_size=opts.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    ### (5) Resume student model & scheduler

    if opts.resume_ckpt is not None and os.path.isfile(opts.resume_ckpt):
        if torch.cuda.device_count() > 1:
            s_model = nn.DataParallel(s_model)
            t_model = nn.DataParallel(t_model)
        checkpoint = torch.load(opts.resume_ckpt, map_location=torch.device('cpu'))
        s_model.load_state_dict(checkpoint["model_state"])
        s_model.to(devices)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            resume_epoch = checkpoint["cur_itrs"]
            print("Training state restored from %s" % opts.resume_ckpt)
        else:
            resume_epoch = 0
        print("Model restored from %s" % opts.resume_ckpt)
        del checkpoint  # free memory
        torch.cuda.empty_cache()
    else:
        print("[!] Train from scratch...")
        resume_epoch = 0
        if torch.cuda.device_count() > 1:
            print('cuda multiple GPUs')
            s_model = nn.DataParallel(s_model)
            t_model = nn.DataParallel(t_model)
        s_model.to(devices)
        t_model.to(devices)

    #### (6) Set up metrics

    metrics = StreamSegMetrics(opts.num_classes)
    early_stopping = utils.EarlyStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.best_ckpt, save_model=opts.save_model)
    dice_stopping = utils.DiceStopping(patience=opts.patience, verbose=True, delta=0.0001,
                                            path=opts.best_ckpt, save_model=opts.save_model)

    ### (7) Train

    B_epoch = 0
    B_test_score = {}

    for epoch in range(resume_epoch, opts.total_itrs):
        score, epoch_loss, lbd, beta = _accumulate(s_model=s_model, t_model=t_model, loader=s_train_loader, 
                                        optimizer=optimizer, scheduler=scheduler, 
                                        get_metrics=True, device=devices,
                                        metrics=metrics, criterion=criterion)
        if epoch > 0:
            for i in range(14):
                print(LINE_UP, end=LINE_CLEAR) 

        print("[{}] Epoch: {}/{} Loss: {:.5f}".format('Train', epoch+1, opts.total_itrs, epoch_loss))
        print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
        print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
        print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))

        writer.add_scalar('IoU BG/train', score['Class IoU'][0], epoch)
        writer.add_scalar('IoU Nerve/train', score['Class IoU'][1], epoch)
        writer.add_scalar('Dice BG/train', score['Class F1'][0], epoch)
        writer.add_scalar('Dice Nerve/train', score['Class F1'][1], epoch)
        writer.add_scalar('epoch loss/train', epoch_loss, epoch)
        writer.add_scalar('CE Lambda/train', lbd, epoch)
        writer.add_scalar('DL Beta/train', beta, epoch)
        
        if (epoch + 1) % opts.val_interval == 0:
            val_score, val_loss = _validate(opts, s_model, t_model, s_val_loader, 
                                            devices, metrics, epoch, criterion)

            print("[{}] Epoch: {}/{} Loss: {:.5f}".format('Val', epoch+1, opts.total_itrs, val_loss))
            print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(val_score['Class F1'][0], val_score['Class F1'][1]))
            print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(val_score['Class IoU'][0], val_score['Class IoU'][1]))
            print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(val_score['Overall Acc'], val_score['Mean Acc']))
            
            early_stopping(val_loss, s_model, optimizer, scheduler, epoch)

            writer.add_scalar('IoU BG/val', val_score['Class IoU'][0], epoch)
            writer.add_scalar('IoU Nerve/val', val_score['Class IoU'][1], epoch)
            writer.add_scalar('Dice BG/val', val_score['Class F1'][0], epoch)
            writer.add_scalar('Dice Nerve/val', val_score['Class F1'][1], epoch)
            writer.add_scalar('epoch loss/val', val_loss, epoch)
        
        if (epoch + 1) % opts.test_interval == 0:
            test_score, test_loss = _validate(opts, s_model, t_model, s_test_loader, 
                                            devices, metrics, epoch, criterion)

            print("[{}] Epoch: {}/{} Loss: {:.5f}".format('Test', epoch+1, opts.total_itrs, test_loss))
            print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(test_score['Class F1'][0], test_score['Class F1'][1]))
            print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(test_score['Class IoU'][0], test_score['Class IoU'][1]))
            print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(test_score['Overall Acc'], test_score['Mean Acc']))

            if dice_stopping(test_score['Class F1'][1], s_model, optimizer, scheduler, epoch):
                B_epoch = epoch
                B_test_score = test_score
        
            writer.add_scalar('IoU BG/test', test_score['Class IoU'][0], epoch)
            writer.add_scalar('IoU Nerve/test', test_score['Class IoU'][1], epoch)
            writer.add_scalar('Dice BG/test', test_score['Class F1'][0], epoch)
            writer.add_scalar('Dice Nerve/test', test_score['Class F1'][1], epoch)
            writer.add_scalar('epoch loss/test', test_loss, epoch)
        
        if early_stopping.early_stop:
            print("Early Stop !!!")
            break
        
        if opts.run_demo and epoch > 5:
            print("Run demo !!!")
            break
    
    if opts.save_test_results:
        params = utils.Params(json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json')).dict
        for k, v in B_test_score.items():
            params[k] = v
        utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))

        if opts.save_model:
            if opts.save_model:
                checkpoint = torch.load(os.path.join(opts.best_ckpt, 'dicecheckpoint.pt'), map_location=devices)
            s_model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.test_results_dir, 'epoch_{}'.format(B_epoch))
            utils.save(sdir, s_model, s_test_loader, devices, opts.is_rgb)
            del checkpoint
            del s_model
            torch.cuda.empty_cache()
        else:
            checkpoint = torch.load(os.path.join(opts.best_ckpt, 'dicecheckpoint.pt'), map_location=devices)
            s_model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.test_results_dir, 'epoch_{}'.format(B_epoch))
            utils.save(sdir, s_model, s_test_loader, devices, opts.is_rgb)
            del checkpoint
            del s_model
            torch.cuda.empty_cache()
            if os.path.exists(os.path.join(opts.best_ckpt, 'checkpoint.pt')):
                os.remove(os.path.join(opts.best_ckpt, 'checkpoint.pt'))
            if os.path.exists(os.path.join(opts.best_ckpt, 'dicecheckpoint.pt')):
                os.remove(os.path.join(opts.best_ckpt, 'dicecheckpoint.pt'))
            os.rmdir(os.path.join(opts.best_ckpt))

    return {
                'Model' : opts.s_model, 'Dataset' : opts.s_dataset,
                'OS' : str(opts.output_stride), 'Epoch' : str(B_epoch),
                'F1 [0]' : B_test_score['Class F1'][0], 'F1 [1]' : B_test_score['Class F1'][1]
            }


        
