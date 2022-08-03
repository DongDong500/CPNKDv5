import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image

import utils

def lpmap(N=2, lbl=True):
    """
    Args:
        N: The number of classes
        lbl: if True, it represents label (Blue) else predicted value (Red)
    """

    cmap = np.zeros((N, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([0, 0, 255]) if lbl else np.array([255, 0, 0])

    return cmap

def save(path, model, loader, device, rgb):
    """
    Args:
        path: Images are saved to the path
        model: Trained model
        loader: 1 epoch of (Train/Val) images to be saved
        device: cuda/cpu
        rgb: (RGB=3channels) = True, (Gray=1channel) = False

    Returns:
        ...
    """
    lmap = lpmap(lbl=True)
    pmap = lpmap(lbl=False)

    if not os.path.exists(path):
        try:
            utils.mkdir(path)
        except:
            raise Exception("Error: can not make directory: {}".format(path))
            
    if rgb:
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        denorm = utils.Denormalize(mean=[0.485], std=[0.229])

    for i, (images, labels) in tqdm(enumerate(loader)):
        '''
            Images shape: B x Channel x H x W 
                Ex) (5, 1, 512, 512)
            Labels shape: B x H x W 
                Ex) (5, 512, 512)
                if multi-class B x Class x H x W
            Output shape: B x Class x H x W
        '''
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        outputs = model(images)
        probs = nn.Softmax(dim=1)(outputs)

        preds = torch.max(probs, 1)[1].detach().cpu().numpy()
        image = images.detach().cpu().numpy()
        lbl = labels.detach().cpu().numpy()
        
        for j in range(images.shape[0]):
            '''
                Denorm(image[j]) shape: H x W x Channel
                    Ex) (512, 512, 1)
                lmap(lbl[j]) shape: H x W x Channel
                    Ex) (512, 512, 3)
            '''
            tar1 = (denorm(image[j]) * 255).transpose(1, 2, 0).astype(np.uint8)
            tar2 = lmap[lbl[j]]
            tar3 = pmap[preds[j]]
            tar4 = tar2 + tar3
            if not rgb:
                # Expand to 3 channels
                tar = np.zeros_like(tar4)
                tar[:,:,0] = tar1
                tar[:,:,1] = tar1
                tar[:,:,2] = tar1
                tar1 = tar
            tar5 = (tar4*0.5 + tar1*0.5).astype(np.uint8)

            if not rgb:
                tar1 = np.squeeze(tar1)

            idx = str(i*images.shape[0] + j).zfill(3)
            Image.fromarray(tar5).save(os.path.join( path, '{}_overlay.png'.format(idx) ))

if __name__ == "__main__":
    import sys
    from os import path
    print(path.dirname( path.dirname( path.abspath(__file__) ) ))
    sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
    import network
    import datasets as dt
    import ext_transforms as et
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    isrgb = False
    separable_conv = True
    model_name = 'deeplabv3plus_resnet101'

    if isrgb:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.485]
        std = [0.229]
    
    val_transform = et.ExtCompose([
        et.ExtResize(size=(496, 468)),
        et.ExtRandomCrop(size=(512, 448), pad_if_needed=True),
        et.ExtScale(scale=0.5),
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std)
        ])
    val_dst = dt.CPN(root='/data1/sdi/datasets', datatype='CPN_six',
                        image_set='train', transform=val_transform,
                        is_rgb=isrgb)
    val_loader = DataLoader(val_dst, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    print("[!] Val set: %d" % (len(val_dst)))

    model = network.model.__dict__[model_name](channel=3 if isrgb else 1, num_classes=2, output_stride=8)
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    model = nn.DataParallel(model)
    model.to(device)

    #for param in model.state_dict():
    #    print(param, "\t", model.state_dict()[param].size())

    model.load_state_dict(torch.load('/data1/sdi/MUnetPlus-result/deeplabv3plus_resnet101/Apr20_22-43-22_CPN_six/best_param/checkpoint.pt'))

    save(path='/data1/sdi/MUnetPlus-result/test0509', model=model, loader=val_loader, device=device, rgb=isrgb)