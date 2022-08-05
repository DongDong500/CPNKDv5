import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import utils
import network
from metrics import StreamSegMetrics
from _get_dataset import get_dataset
from _validate import validate
from _train import train

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def _load_model(opts = None, verbose = True, pretrain = False):
    
    print("<load model> %s" % opts.model) if verbose else 0 

    if opts.model.startswith("deeplabv3plus"):
        model = network.model.__dict__[opts.model](in_channel=opts.in_channels, 
                                                    classes=opts.classes,
                                                    encoder_name=opts.encoder_name,
                                                    encoder_depth=opts.encoder_depth,
                                                    encoder_weights=opts.encoder_weights,
                                                    encoder_output_stride=opts.encoder_output_stride,
                                                    decoder_atrous_rates=opts.decoder_atrous_rates,
                                                    decoder_channels=opts.decoder_channels,
                                                    activation=opts.activation,
                                                    upsampling=opts.upsampling,
                                                    aux_params=opts.aux_params)
    else:
        model = network.model.__dict__[opts.model](channel=opts.in_channels, 
                                                    num_classes=opts.classes)

    if pretrain and os.path.isfile(opts.model_params):
        print("<load model> restored parameters from %s" % opts.model_params) if verbose else 0
        checkpoint = torch.load(opts.model_params, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        del checkpoint  # free memory
        torch.cuda.empty_cache()

    return model

def experiments(opts, run_id) -> dict:

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (devices, opts.gpus))
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    logdir = os.path.join(opts.Tlog_dir, 'run_' + str(run_id).zfill(2))
    writer = SummaryWriter(log_dir=logdir) 

    ### (1) Get datasets

    train_dst, val_dst, test_dst = get_dataset(opts, opts.dataset, opts.dataset_ver)
    
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dst, batch_size=opts.test_batch_size, num_workers=opts.num_workers, 
                                shuffle=True, drop_last=True)

    ### (2) Set up criterion
    '''
        depreciated
    '''

    ### (3 -1) Load teacher & student models

    model = _load_model(opts=opts, verbose=True)

    ### (4) Set up optimizer

    if opts.model.startswith("deeplab"):
        if opts.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
        elif opts.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError
    else:
        optimizer = optim.RMSprop(model.parameters(), 
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

    if opts.resume and os.path.isfile(opts.resume_ckpt):
        checkpoint = torch.load(opts.resume_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(devices)
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
        model = nn.DataParallel(model)

    model.to(devices)

    #### (6) Set up metrics

    metrics = StreamSegMetrics(opts.classes)
    early_stopping = utils.EarlyStopping(patience=opts.patience, verbose=True, delta=opts.delta,
                                            path=opts.best_ckpt, save_model=opts.save_model)
    dice_stopping = utils.DiceStopping(patience=opts.patience, verbose=True, delta=0.0001,
                                            path=opts.best_ckpt, save_model=opts.save_model)

    ### (7) Train

    B_epoch = 0
    B_test_score = {}

    for epoch in range(resume_epoch, opts.total_itrs):
        score, epoch_loss = train(model=model, loader=train_loader, devices=devices, metrics=metrics, 
                                    loss_type=opts.loss_type, optimizer=optimizer, scheduler=scheduler, opts=opts)
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
        
        if (epoch + 1) % opts.val_interval == 0:
            val_score, val_loss = validate(model=model, loader=val_loader, devices=devices, 
                                                metrics=metrics, loss_type='dice_loss')

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
            test_score, test_loss = validate(model=model, loader=val_loader, devices=devices, 
                                                metrics=metrics, loss_type='dice_loss')

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
            utils.save(sdir, model, test_loader, devices, opts.is_rgb)
            del checkpoint
            del s_model
            torch.cuda.empty_cache()
        else:
            checkpoint = torch.load(os.path.join(opts.best_ckpt, 'dicecheckpoint.pt'), map_location=devices)
            s_model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.test_results_dir, 'epoch_{}'.format(B_epoch))
            utils.save(sdir, model, test_loader, devices, opts.is_rgb)
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


        
