import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vit_pytorch import ViT

from utils import ext_transforms as et
from datasets.cpn_vit import CPNvit
from metrics import ClsMetrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0,
                        help="gpus (default: 0)")
    parser.add_argument("--dir", type=str, default="/data1/sdi/CPNKDv5-result/vit/run_13",
                        help="weights path (~/vit/run_00)")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    torch.cuda.set_device(args.gpus)
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'devices: {devices}')

    if not os.path.exists(os.path.join(args.dir, 'checkpoint.pt')):
        raise Exception ("File not found: ", args.dir)

    if not os.path.exists(os.path.join(args.dir, 'inference')):
        os.mkdir(os.path.join(args.dir, 'inference'))

    v = ViT(
            image_size = 512,
            patch_size = 16,
            num_classes = 2,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
            )
    ckpt = torch.load(os.path.join(args.dir, 'checkpoint.pt'), map_location='cpu')
    v.load_state_dict(ckpt["model_state"])
    v.to(devices)

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), is_crop=True, pad_if_needed=True),
            #et.ExtResize(size=(224, 224), is_resize=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    val_dst = CPNvit(root='/data1/sdi/datasets', datatype='CPN', image_set='test',
                transform=transform, is_rgb=True, dver='splits/v5/3')
    val_loader = DataLoader(val_dst, batch_size=8,
                            shuffle=True, num_workers=2, drop_last=True)
    metric = ClsMetrics(n_classes=2)
    
    v.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(devices)
            labels = labels.to(devices)

            outputs = v(images)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            y_pred = preds.detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            metric.update(y_true, y_pred)

        score = metric.get_results()
    print("----------------------- Matrix -----------------------")
    print(metric.confusion_matrix)
    print(f"best epoch: {ckpt['cur_itrs']}")
    print("----------------------- Report -----------------------")
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))
    print("----------------------- ------ -----------------------")