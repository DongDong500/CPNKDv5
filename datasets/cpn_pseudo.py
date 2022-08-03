import os
import sys
import random
import torch.utils.data as data
from PIL import Image

class CPNPseudo(data.Dataset):
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
    def _read(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError
        
        if self.is_rgb:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L')         
        else:
            img = Image.open(self.images[index]).convert('L')
            target = Image.open(self.masks[index]).convert('L')            

        return img, target

    def __init__(self, root, datatype='CPN', dver='splits', 
                    image_set='train', transform=None, is_rgb=True, ratio=0.5, **kwargs):
        if ratio >= 1:
            raise Exception ('ratio must be smaller than 1.0')

        self.transform = transform
        self.is_rgb = is_rgb

        image_dir = os.path.join(root, 'CPN_all', 'Images')
        mask_dir = os.path.join(root, 'CPN_all', 'Masks')
        p_image_dir = os.path.join(root, 'CPN_all', 'Images')
        p_mask_dir = os.path.join(root, 'CPN_pseudo', 'Masks')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise Exception('Dataset not found or corrupted.')
        if not os.path.exists(p_mask_dir):
            raise Exception('Dataset not found or corrupted.')

        split_f = os.path.join(root, 'CPN_all', dver, image_set.rstrip('\n') + '.txt')
        p_split_f = os.path.join(root, 'CPN_pseudo', dver, image_set.rstrip('\n') + '.txt')
        
        if not os.path.exists(split_f) or not os.path.exists(p_split_f):
            raise Exception('Wrong image_set entered!' , split_f, p_split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        with open(os.path.join(p_split_f), "r") as f:
            p_file_names = [x.strip() for x in f.readlines()]

        r_index = int(random.uniform(0, len(p_file_names)-1))
        r_index_len = int(len(p_file_names) * ratio) + 1
        if r_index + r_index_len > len(p_file_names):
            p_file_names = p_file_names[r_index:] + p_file_names[:r_index_len + r_index - len(p_file_names)]
        else:
            p_file_names = p_file_names[r_index:r_index + r_index_len]

        self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]

        if image_set == 'train' or image_set == 'val':
            
            self.images.extend([os.path.join(p_image_dir, x + ".bmp") for x in p_file_names])
            self.masks.extend([os.path.join(p_mask_dir, x + "_mask.bmp") for x in p_file_names])

        assert (len(self.images) == len(self.masks))

        self.image = []
        self.mask = []
        for index in range(len(self.images)):
            img, tar = self._read(index)
            self.image.append(img)
            self.mask.append(tar)

    def __getitem__(self, index):

        img = self.image[index]
        target = self.mask[index]
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target

    def __len__(self):
        return len(self.images)


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
    
    dst = CPNPseudo(root='/data1/sdi/datasets', datatype='CPN', image_set='test',
                    transform=transform, is_rgb=True, dver='splits/v5/3', ratio=0.0)
    train_loader = DataLoader(dst, batch_size=16,
                                shuffle=True, num_workers=2, drop_last=True)
    
    for i, (ims, lbls) in tqdm(enumerate(train_loader)):
        print(ims.shape)
        print(lbls.shape)
        print(lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
        print(1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
        if i > 1:
            break
    