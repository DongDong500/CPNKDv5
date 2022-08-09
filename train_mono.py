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
import criterion
from metrics import StreamSegMetrics
from _get_dataset import get_dataset
# from _validate import validate
# from _train import train

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'


def add_writer_scalar(writer, phase, score, loss, epoch):

    writer.add_scalar(f'IoU BG/{phase}', score['Class IoU'][0], epoch)
    writer.add_scalar(f'IoU Nerve/{phase}', score['Class IoU'][1], epoch)
    writer.add_scalar(f'Dice BG/{phase}', score['Class F1'][0], epoch)
    writer.add_scalar(f'Dice Nerve/{phase}', score['Class F1'][1], epoch)
    writer.add_scalar(f'epoch loss/{phase}', loss, epoch)

def print_result(phase, score, epoch, total_itrs, loss):

    print("[{}] Epoch: {}/{} Loss: {:.5f}".format(phase, epoch, total_itrs, loss))
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))

def validate(model, loader, devices, metrics, loss_type,**kwargs):
    
    costfunction = criterion.get_criterion.__dict__[loss_type](**kwargs)

    model.eval()
    metrics.reset()

    running_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(devices)
            labels = labels.to(devices)

            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = labels.detach().cpu().numpy()

            loss = costfunction(outputs, labels)

            metrics.update(target, preds)
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()

    return score, epoch_loss


def get_class_weight(lbls):
    '''
    compute class weight (only binary)
        Args:
            lbls (numpy array)
        Returns:
            weight (numpy array)
    '''
    weights = lbls.sum() / (lbls.shape[0] * lbls.shape[1] * lbls.shape[2])
    
    if weights < 0 or weights > 1:
        raise Exception (f'weights: {weights} for cross entropy is wrong')

    return [weights, 1 - weights]

def train(model, loader, devices, metrics, loss_type, 
            optimizer, scheduler, **kwargs):
    
    costfunction = criterion.get_criterion.__dict__[loss_type](**kwargs)

    model.train()
    metrics.reset()
    running_loss = 0.0

    for i, (images, labels) in enumerate(loader):
        images = images.to(devices)
        labels = labels.to(devices)
        
        outputs = model(images)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1].detach().cpu().numpy()

        if loss_type == 'entropydice_loss':
            class_weights = torch.tensor(get_class_weight(labels.detach().cpu().numpy()), dtype=torch.float32).to(devices)
            costfunction.update_weight(weight=class_weights)
        elif loss_type == 'dice_loss':
            pass
        elif loss_type == 'kd_loss':
            raise NotImplementedError
        elif loss_type == 'gp_loss':
            raise NotImplementedError
        else:
            raise Exception (f'{loss_type} is not option')

        optimizer.zero_grad()
        loss = costfunction(outputs, labels)
        loss.backward()

        optimizer.step()
        
        metrics.update(labels.detach().cpu().numpy(), preds)
        running_loss += loss.item() * images.size(0)
    
    scheduler.step()
    epoch_loss = running_loss / len(loader.dataset)
    score = metrics.get_results()
    
    return score, epoch_loss

def _load_model(opts = None, verbose = True, pretrain = False):
    
    print("<load model> %s" % opts.model) if verbose else 0 

    if opts.model.startswith("deeplabv3plus"):
        model = network.model.__dict__[opts.model](in_channels=opts.in_channels, 
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
        raise NotImplementedError

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

    RUN_ID = 'run_' + str(run_id).zfill(2)
    os.mkdir(os.path.join(opts.Tlog_dir, RUN_ID))
    os.mkdir(os.path.join(opts.best_ckpt, RUN_ID))
    os.mkdir(os.path.join(opts.test_results_dir, RUN_ID))
    writer = SummaryWriter(log_dir=os.path.join(opts.Tlog_dir, RUN_ID))

    ### (1) Get datasets
    train_dst, val_dst, test_dst = get_dataset(opts, opts.dataset, opts.dataset_ver)
    
    train_loader = DataLoader(train_dst, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dst, batch_size=opts.val_batch_size, num_workers=opts.num_workers,
                                shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dst, batch_size=opts.test_batch_size, num_workers=opts.num_workers, 
                                shuffle=True, drop_last=True)

    ### (2) Set up criterion
    ''' depreciated
    '''

    ### (3 -1) Load teacher & student models
    model = _load_model(opts=opts, verbose=True)

    ### (4) Set up optimizer
    if opts.model.startswith("deeplab"):
        if opts.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': model.encoder.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.decoder.parameters(), 'lr': opts.lr},
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
    best_epoch = 0
    best_score = 0
    metrics = StreamSegMetrics(opts.classes)
    early_stopping = utils.DiceStopping(patience=opts.patience, delta=opts.delta, verbose=True,
                                                path=os.path.join(opts.best_ckpt, RUN_ID))

    ### (7) Train
    for epoch in range(resume_epoch, opts.total_itrs):
        score, epoch_loss = train(model=model, loader=train_loader, devices=devices, metrics=metrics, 
                                    loss_type=opts.loss_type, optimizer=optimizer, scheduler=scheduler, opts=opts)
        
        # if epoch > 0:
        #     for i in range(14):
        #         print(LINE_UP, end=LINE_CLEAR) 

        print_result('train', score, epoch, opts.total_itrs, epoch_loss)
        add_writer_scalar(writer, 'train', score, epoch_loss, epoch)
        
        # Validate
        val_score, val_loss = validate(model=model, loader=val_loader, devices=devices, 
                                            metrics=metrics, loss_type='dice_loss')
        print_result('val', val_score, epoch, opts.total_itrs, val_loss)
        add_writer_scalar(writer, 'val', val_score, val_loss, epoch)

        # Test
        test_score, test_loss = validate(model=model, loader=test_loader, devices=devices, 
                                            metrics=metrics, loss_type='dice_loss')
        print_result('test', test_score, epoch, opts.total_itrs, test_loss)
        add_writer_scalar(writer, 'test', test_score, test_loss, epoch)

        if early_stopping(test_score['Class F1'][1], model, optimizer, scheduler, epoch):
            best_epoch = epoch
            best_score = test_score
    
        if early_stopping.early_stop:
            print("Early Stop !!!")
            break

        if opts.run_demo and epoch > 5:
            print("Run demo !!!")
            break

    ### (8) Save results
    if opts.save_test_results:
        params = utils.Params(json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json')).dict
        params[f'{RUN_ID} dice score'] = best_score["Class F1"][1]
        utils.save_dict_to_json(d=params, json_path=os.path.join(opts.default_prefix, opts.current_time, 'summary.json'))

        if opts.save_model:
            checkpoint = torch.load(os.path.join(opts.best_ckpt, RUN_ID, 'checkpoint.pt'), map_location=devices)
            model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.test_results_dir, RUN_ID, f'epoch_{best_epoch}')
            utils.save(sdir, model, test_loader, devices, (opts.in_channels == 3))
        else:
            checkpoint = torch.load(os.path.join(opts.best_ckpt, RUN_ID, 'checkpoint.pt'), map_location=devices)
            model.load_state_dict(checkpoint["model_state"])
            sdir = os.path.join(opts.test_results_dir, RUN_ID, f'epoch_{best_epoch}')
            utils.save(sdir, model, test_loader, devices, (opts.in_channels == 3))
            os.remove(os.path.join(opts.best_ckpt, RUN_ID, 'checkpoint.pt'))
            os.removedirs(os.path.join(opts.best_ckpt))
    
    del checkpoint
    del model
    torch.cuda.empty_cache()

    return {
                'Model' : opts.model, 'Dataset' : opts.dataset,
                'OS' : str(opts.encoder_output_stride), 'Epoch' : str(best_epoch),
                'F1 [0]' : best_score['Class F1'][0], 'F1 [1]' : best_score['Class F1'][1]
            }


        
