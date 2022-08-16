import os
import argparse
import socket
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import network
import utils
import criterion
import datasets as dt
from metrics import StreamSegMetrics
from utils import ext_transforms as et
from _get_dataset import get_dataset
# from _validate import validate
from train_mono import validate

DEFAULT_PREFIX = {
    "server2" : "/mnt/server5/sdi",
    "server3" : "/mnt/server5/sdi",
    "server4" : "/mnt/server5/sdi",
    "server5" : "/data1/sdi"
}

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, default='Aug09_03-37-20',
                        help='path without prefix')
    parser.add_argument("--ver", type=str, default='run_00',
                        help='version of run')

    return parser.parse_args()

def get_parser(verbose=True):
    
    args = get_argparser()
    args.pth = os.path.join(DEFAULT_PREFIX[socket.gethostname()], args.pth, 'summary.json')
    if not os.path.exists(args.pth):
        raise Exception (f'pth not exists: {args.pth}')

    pram = utils.Params(args.pth)
    if verbose:
        print(f'folder: {os.path.dirname(args.pth)}')

    return pram, args.ver

def load_model(opts, ver='run_00', ckpt='dicecheckpoint.pt'):

    if opts.model.startswith("deeplabv3plus"):
        model = network.model.__dict__[opts.model](in_channels=opts.in_channels, classes=opts.classes,
                                                    encoder_name=opts.encoder_name, encoder_depth=opts.encoder_depth, 
                                                    encoder_weights=opts.encoder_weights, encoder_output_stride=opts.encoder_output_stride,
                                                    decoder_atrous_rates=opts.decoder_atrous_rates, decoder_channels=opts.decoder_channels,
                                                    activation=opts.activation, upsampling=opts.upsampling, aux_params=opts.aux_params)
        if ver.startswith('run'):  
            ckpt = torch.load(os.path.join(opts.best_ckpt, ver, ckpt), map_location='cpu')
        else:
            ckpt = torch.load(os.path.join(opts.best_ckpt, ckpt), map_location='cpu')

        model.load_state_dict(ckpt["model_state"])
        print(f'Best epoch: { ckpt["cur_itrs"] }')
        del ckpt
    else:
        raise NotImplementedError
    
    return model

def __get_dataset(opts):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if opts.is_gaussian_crop:
        transform = et.ExtCompose([
            et.ExtResize(size=opts.resize_test, is_resize=opts.is_resize_test),
            et.ExtGaussianRandomCrop(size=opts.crop_size_test, 
                                        normal_h=opts.gaussian_crop_H, 
                                        normal_w=opts.gaussian_crop_W,
                                        block_size=opts.gaussian_crop_block_size),
            et.ExtScale(scale=opts.scale_factor_test, is_scale=opts.is_scale_test),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            et.GaussianPerturb(mean=opts.mu_test, std=opts.std_test)
            ])
    else:
        transform = et.ExtCompose([
            et.ExtResize(size=opts.resize_test, is_resize=opts.is_resize_test),
            et.ExtRandomCrop(size=opts.crop_size_test, is_crop=opts.is_crop_test, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor_test, is_scale=opts.is_scale_test),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            et.GaussianPerturb(mean=opts.mu_test, std=opts.std_test)
            ])

    dst = dt.getdata.__dict__[opts.dataset](root=opts.data_root, 
                                            datatype=opts.dataset, 
                                            dver=opts.dataset_ver, 
                                            image_set='test', 
                                            transform=transform, 
                                            is_rgb=True)

    print(f'Dataset: {opts.dataset}/{opts.dataset_ver}, len: {len(dst)}')

    return dst

def fit(metrics, loader, model):

    # EVAL 꼭 해줘야한다 개시발 좆같네 시발
    model.eval()
    with torch.no_grad():
        metrics.reset()
        for i, (img, lbl) in enumerate(loader):

            img = img.to(device)
            lbl = lbl.to(device)

            outputs = model(img)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            target = lbl.detach().cpu().numpy()

            metrics.update(target, preds)
    
    score = metrics.get_results()

    print("----------------------- Report -----------------------")
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))
    print("----------------------- ------ -----------------------")


if __name__ == "__main__":

    opts, ver = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s: %s" % (device, opts.gpus))
    
    metrics = StreamSegMetrics(opts.classes)
    metrics.reset()
    model = load_model(opts, ver, ckpt='checkpoint.pt').to(device)
    # _, _, dst = get_dataset(opts, opts.dataset, opts.dataset_ver)
    dst = __get_dataset(opts)

    loader = DataLoader(dst, batch_size=opts.test_batch_size, num_workers=4,
                        shuffle=True, drop_last=True)

    fit(metrics=metrics, loader=loader, model=model)

    score, _ = validate(model=model, loader=loader, devices=device, 
                            metrics=metrics, loss_type='dice_loss')
    score = metrics.get_results()
    print("----------------------- Report -----------------------")
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))
    print("----------------------- ------ -----------------------")

    