import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from vit_pytorch import ViT

from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torchvision import models
from torchvision.models import ResNet101_Weights

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

import utils
import criterion
from datasets.cpn_vit import CPNvit
from utils import ext_transforms as et
from metrics import ClsMetrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0,
                        help="gpus (default: 0)")
    parser.add_argument("--log_dir", type=str, default="/data1/sdi/CPNKDv5-result/vit",
                        help="tensorboard log path")
    return parser.parse_args()

def add_writer_scalar(writer, phase, score, loss, epoch, num_classes):
    
    writer.add_scalar(f'vit32/epoch loss/cls {num_classes}/{phase}', loss, epoch)
    writer.add_scalar(f'vit32/epoch Overall Acc/cls {num_classes}/{phase}', score['Overall Acc'], epoch)
    
    for i in range(num_classes):
        writer.add_scalar(f'vit32/f1 score/cls {num_classes}/{i}/{phase}', score['Class F1'][i], epoch)

    print(f'[{phase}]\t {epoch+1}/2000 Overall Acc: {score["Overall Acc"]:.4f}, loss: {loss:.4f}')


def vit(image_patch_size:tuple, vit_patch_size:int):
    
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args = get_args()

    torch.cuda.set_device(args.gpus)
    run_id = 1 + int(sorted(os.listdir(args.log_dir))[-1].split('_')[-1])
    run_id = 'run_' + str(run_id).zfill(2)
    os.mkdir(os.path.join(args.log_dir, run_id))
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_id))

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'devices: {devices}')
    
    patch_size = image_patch_size
    num_cls = (512 // patch_size[0]) * (512 // patch_size[1])
    print(f'number of classes: {num_cls}')
    
    v = ViT(
            image_size = 512,
            patch_size = vit_patch_size,
            num_classes = num_cls,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            ).to(devices)

    #v = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True, )
    # v = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    # num_ftrs = v.fc.in_features
    # v.fc = nn.Linear(num_ftrs, 2)
    # v.to(devices)

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), is_crop=True, pad_if_needed=True),
            #et.ExtResize(size=(224, 224), is_resize=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    dst = CPNvit(root='/mnt/server5/sdi/datasets', datatype='CPN_vit', image_set='train',
                transform=transform, is_rgb=True, dver='splits/v5/3', image_patch_size=patch_size)
    loader = DataLoader(dst, batch_size=8,
                            shuffle=True, num_workers=2, drop_last=True)

    val_dst = CPNvit(root='/mnt/server5/sdi/datasets', datatype='CPN_vit', image_set='test',
                transform=transform, is_rgb=True, dver='splits/v5/3', image_patch_size=patch_size)
    val_loader = DataLoader(val_dst, batch_size=8,
                            shuffle=True, num_workers=2, drop_last=True)

    print(f'[train]: {len(dst)}, [test]: {len(val_dst)}')

    optimizer = optim.SGD(v.parameters(), 
                            lr=5e-4,
                            weight_decay=5e-4,
                            momentum=0.9)
    scheduler = utils.PolyLR(optimizer, 2000, power=0.9)

    #costfunction = CrossEntropyLoss(weight=torch.tensor([80/374, 290/374]).to(devices))
    # costfunction = MSELoss()
    # weight = [1 for i in range(num_cls)]
    # costfunction = CrossEntropyLoss(weight=torch.tensor(weight).to(devices))
    
    costfunction = CrossEntropyLoss()
    metric = ClsMetrics(n_classes=num_cls)
    early_stopping = utils.EarlyStopping(patience=200, delta=1e-4, verbose=True,
                                            path=os.path.join(args.log_dir, run_id))

    for epoch in range(0, 2000):

        v.train()
        metric.reset()
        running_loss = 0.0
        for i, (images, labels) in enumerate(loader):
            labels = labels.type(torch.long)
            images = images.to(devices)
            labels = labels.to(devices)

            outputs = v(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            optimizer.zero_grad()
            loss = costfunction(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            y_pred = preds.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            metric.update(y_true, y_pred)

        scheduler.step()
        epoch_loss = running_loss / len(loader.dataset)
        score = metric.get_results()
        add_writer_scalar(writer, 'train', score, epoch_loss, epoch, num_cls)

        v.eval()
        metric.reset()
        running_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                labels = labels.type(torch.long)
                images = images.to(devices)
                labels = labels.to(devices)

                outputs = v(images)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]

                loss = costfunction(outputs, labels)
                running_loss += loss.item() * images.size(0)               

                y_pred = preds.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy()
                metric.update(y_true, y_pred)

            epoch_loss = running_loss / len(val_loader.dataset)
            score = metric.get_results()
            add_writer_scalar(writer, 'val', score, epoch_loss, epoch, num_cls)

            early_stopping(epoch_loss, v, optimizer, scheduler, epoch)
            if early_stopping.early_stop:
                print("Early Stop !!!")
                break
    
    
if __name__ == "__main__":
    '''
    vit patch size = 32
    '''

    #cls = [(256, 256), (128, 128), (64, 64)]
    cls = [(32, 32)]
    run_time = []

    for i in cls:
        start_time = datetime.now()
        vit(image_patch_size=i, vit_patch_size=32)
        run_time.append( datetime.now() - start_time )

    for i in run_time:
        print(f'time elapsed: {i}')