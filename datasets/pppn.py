import os
import sys
import numpy as np
import torch.utils.data as data
from PIL import Image

class PPPN(data.Dataset):
    """
    Args:6
        root (string): Root directory of the VOC Dataset.
        datatype (string): Dataset type 
        image_set (string): Select the image_set to use, ``train``, ``val`` or ``test``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        dver (str): version of dataset (ex) ``splits/v5/3``
        kfold (int): k-fold cross validation
    """
    def __init__(self, root, datatype='CPN', dver='splits', 
                    image_set='train', transform=None, is_rgb=True):

        self.transform = transform
        self.is_rgb = is_rgb

        if image_set == 'train' or image_set == 'val':
            self.is_test = False
        else:
            self.is_test = True

        image_dir = os.path.join(root, 'CPN_all', 'Images')
        mask_dir = os.path.join(root, 'CPN_all', 'Masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise Exception('Dataset not found or corrupted.')
        
        split_f = os.path.join(root, 'CPN_all', dver, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise Exception('Wrong image_set entered!', split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]
        
        assert (len(self.images) == len(self.masks))      

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        r_index = np.random.randint(0, len(self.images))

        if not os.path.exists(self.images[index]) or not os.path.exists(self.images[r_index]):
            raise FileNotFoundError
        if not os.path.exists(self.masks[index]) or not os.path.exists(self.masks[r_index]):
            raise FileNotFoundError

        if self.is_test:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L') 
        else:
            img = Image.open(self.images[index]).convert('L')
            r_img = Image.open(self.images[r_index]).convert('L')
            r_img = r_img.resize(img.size)
            target = Image.open(self.masks[index]).convert('L')            
            img = np.expand_dims(np.array(img, dtype='uint8'), axis=2)
            r_img = np.expand_dims(np.array(r_img, dtype='uint8'), axis=2)
            img = Image.fromarray(np.concatenate((img, r_img, img), axis=2))
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target


if __name__ == "__main__":

    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.ExtScale(scale=0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    dst = PPPN(root='/data1/sdi/datasets', datatype='pgpn', image_set='train',
                    transform=transform, is_rgb=True, dver='splits/v5/3')
    train_loader = DataLoader(dst, batch_size=16,
                                shuffle=True, num_workers=2, drop_last=True)
    print(f'dataset len(dst) = {len(dst)}')
    for i, (ims, lbls) in tqdm(enumerate(train_loader)):
        print(ims.shape)
        print(lbls.shape)
        print(lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
        print(1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
        if i > 1:
            break
    