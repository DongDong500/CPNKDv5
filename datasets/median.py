import os
import sys
import torch.utils.data as data
from PIL import Image

class Median(data.Dataset):
    """
    Args:6
        root (string): Root directory of the VOC Dataset.
        datatype (string): Dataset type 
        image_set (string): Select the image_set to use, ``train``, ``val`` or ``test``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        dver (str): version of dataset (ex) ``splits``
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

        assert( img.size == target.size == (512, 512) )

        return img, target

    def __init__(self, root, datatype='Median', dver='splits', 
                    image_set='train', transform=None, is_rgb=True):

        self.root = root
        self.datatype = datatype
        self.dver = dver
        self.image_set = image_set
        self.transform = transform
        self.is_rgb = is_rgb

        image_dir = os.path.join(self.root, self.datatype, 'Images')
        mask_dir = os.path.join(self.root, self.datatype, 'Masks')
        split_f = os.path.join(self.root, self.datatype, self.dver, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise Exception('Dataset not found or corrupted.')
        
        if not os.path.exists(split_f):
            raise Exception('Wrong image_set entered!' 
                            'Please use image_set="train" or image_set="val"', split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".jpg") for x in file_names]
        
        assert (len(self.images) == len(self.masks))

        self.image = []
        self.mask = []
        for index in range(len(self.images)):
            img, tar = self._read(index)
            self.image.append(img)
            self.mask.append(tar)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
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
            et.ExtRandomCrop(size=(512, 512), is_crop=True, pad_if_needed=True),
            et.ExtScale(scale=0.5, is_scale=True),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    image_set_type = ['train', 'val', 'test']
    for ist in image_set_type:
        dst = Median(root='/data1/sdi/datasets', datatype='Median', image_set=ist,
                    transform=transform, is_rgb=True, dver='splits')
        loader = DataLoader(dst, batch_size=16,
                                shuffle=True, num_workers=2, drop_last=True)
        print(f'len [{ist}]: {len(dst)}')

        for i, (ims, lbls) in tqdm(enumerate(loader)):
            pass
        
        print('Clear !!!')
    