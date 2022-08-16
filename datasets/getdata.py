import os
import sys
import math
from random import sample
#sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
from .cpn import CPN
from .cpn_vit import CPNvit
from .pmn import PMN
from .median import Median
from .medianpad import Medianpad
from .medianpadw import Medianpadw
from .pgn import PGN
from .pgmn import PGMN
from .pgpn import PGPN
from .pppn import PPPN
from .cpn_pseudo import CPNPseudo
from .cpn_trim import CPNtrim
from .cpn_grc_trim_test import CPNwithTrimTest
from .pmpn import PMPN
from .cpnBypppn import _PPPN
from .cpnpad import CPNpad
from .cpnpadw import CPNpadw
from utils.ext_transforms import ExtCompose

def mktv(root:str = '/', datatype:str = 'CPN', dver:str = 'splits/v5/3', tvs:int = 5):
    split_f = os.path.join(root, datatype, dver, '_train.txt')

    if not os.path.exists(split_f):
        raise Exception(f'_train.txt not found or corrupted. {split_f}')
    
    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    n = math.floor(len(file_names) / tvs)

    val = sample(file_names, n)
    train = list(set(file_names) - set(val))

    with open(os.path.join(root, datatype, dver, 'train.txt'), 'w') as f:
        for w in train:
            f.write(f'{w}\n')
    with open(os.path.join(root, datatype, dver, 'val.txt'), 'w') as f:
        for w in val:
            f.write(f'{w}\n')
    
def cpn(root:str = '/', datatype:str = 'CPN', dver:str = 'splits/v5/3',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
    Args:
        root (str)  :   path to data parent directory (Ex: /data1/sdi/datasets) 
        datatype (str)  :   data folder name (default: CPN_all)
        dver (str)  : version of dataset (default: splits)
        image_set (str) :    train/val or test (default: train)
        transform (ExtCompose)  :   composition of transform class
        is_rgb (bool)   :  True for RGB, False for gray scale images
        tvs (int)   :  train/validate dataset ratio 
                2 block = 1 mini-block train set, 1 mini-block validate set
                5 block = 4 mini-block train set, 1 mini-block validate set
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN', dver, tvs)

    return CPN(root, 'CPN' ,dver, image_set, transform, is_rgb)

def median(root:str = '/', datatype:str = 'Median', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ - Median nerve:     1305 samples (1044 + 261)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'Median', 'splits', tvs)

    return Median(root, 'Median', dver, image_set, transform, is_rgb)

def pmn(root:str = '/', datatype:str = 'PMN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ ```in batch training``` 

        - Peroneal nerve:   490 samples
        - Median nerve:     1305 samples
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN', dver, tvs)
        mktv(root, 'Median', 'splits',tvs)

    return PMN(root, 'CPN', dver, image_set, transform, is_rgb)



def cpnpseudo(root:str = '/', datatype:str = 'CPN', dver:str = 'splits',
                image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
                **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -cpn + cpn pseudo label
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)
        mktv(root, 'CPN_pseudo', dver, tvs)

    return CPNPseudo(root, 'CPN_all', dver, image_set, transform, is_rgb, **kwargs)

def cpntrim(root:str = '/', datatype:str = 'CPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples trimed (256, 256)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_trim', dver, tvs)

    return CPNtrim(root, 'CPN_trim', dver, image_set, transform, is_rgb)

def cpnwithtrimtest(root:str = '/', datatype:str = 'CPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples trimed (256, 256)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)

    return CPNwithTrimTest(root, 'CPN_all', dver, image_set, transform, is_rgb)

def medianpad(root:str = '/', datatype:str = 'Median_pad', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Median nerve
        1044 + 261 = 1305 samples
        size: (896, 640) 501 samples or (640, 640) 802 samples
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'Median_pad', 'splits',tvs)

    return Medianpad(root, 'Median_pad', dver, image_set, transform, is_rgb)

def medianpadw(root:str = '/', datatype:str = 'Median_pad', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Median nerve
        1044 + 261 = 1305 samples
        size: (896, 640) 501 samples or (640, 640) 802 samples
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'Median_padw', 'splits',tvs)

    return Medianpadw(root, 'Median_padw', dver, image_set, transform, is_rgb)

def pgn(root:str = '/', datatype:str = 'PGN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -Gaussian Mixture
    
    Args:
        root (str)  :   path to data parent directory (Ex: /data1/sdi/datasets) 
        datatype (str)  :   data folder name (default: CPN_all)
        dver (str)  : version of dataset (default: splits)
        image_set (str) :    train/val or test (default: train)
        transform (ExtCompose)  :   composition of transform class
        is_rgb (bool)   :  True for RGB, False for gray scale images
        tvs (int)   :  train/validate dataset ratio 
                2 block = 1 mini-block train set, 1 mini-block validate set
                5 block = 4 mini-block train set, 1 mini-block validate set
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)
        mktv(root, 'CPN_all_gmm/1sigma', dver, tvs)

    return PGN(root, 'CPN_all', dver, image_set, transform, is_rgb)

def pgmn(root:str = '/', datatype:str = 'PGMN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -Median nerve
        1044 + 261 = 1305 samples
        -3 channel concatenate(CPN + gaussian random channel + Median)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)
        mktv(root, 'Median', 'splits',tvs)

    return PGMN(root, 'CPN_all', dver, image_set, transform, is_rgb, **kwargs)

def pmpn(root:str = '/', datatype:str = 'PMPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -Median nerve
        1044 + 261 = 1305 samples
        -3 channel concatenate(CPN + Median + CPN)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)
        mktv(root, 'Median', 'splits',tvs)

    return PMPN(root, 'CPN_all', dver, image_set, transform, is_rgb)

def pgpn(root:str = '/', datatype:str = 'PGPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -3 channel concatenate(CPN + gaussian random channel + CPN)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)

    return PGPN(root, 'CPN_all', dver, image_set, transform, is_rgb, **kwargs)

def pppn(root:str = '/', datatype:str = 'PPPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        -3 channel concatenate(CPN + random CPN + CPN)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)

    return PPPN(root, 'CPN_all', dver, image_set, transform, is_rgb)

def cpnbypppn(root:str = '/', datatype:str = 'PPPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -(for validate code) 3 channel concatenate(CPN + CPN + CPN)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_all', dver, tvs)

    return _PPPN(root, 'CPN_all', dver, image_set, transform, is_rgb)

def cpnpad(root:str = '/', datatype:str = 'CPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        with 0 padding (640, 640)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_pad', dver, tvs)

    return CPNpad(root, 'CPN_pad', dver, image_set, transform, is_rgb)

def cpnpadw(root:str = '/', datatype:str = 'CPN', dver:str = 'splits',
            image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True, tvs:int = 5,
            **kwargs):
    
    """ -Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
        with 255 padding (640, 640)
    """
    if tvs < 2:
        raise Exception("tvs must be larger than 1")
    elif image_set == 'train':
        mktv(root, 'CPN_pad_255', dver, tvs)

    return CPNpadw(root, 'CPN_pad_255', dver, image_set, transform, is_rgb)



if __name__ == "__main__":
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.ExtScale(scale=0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    dst = pgn(root='/data1/sdi/datasets', dver='splits/v5/3',
                image_set='train', transform=transform, is_rgb=True, tvs=2)

    loader = DataLoader(dst, batch_size=16,
                        shuffle=True, num_workers=4, drop_last=True)
    print(f'dataset len(dst) = {len(dst)}')
    for i, (ims, lbls) in (enumerate(loader)):
        print(f'image shape: {ims.shape}')
        print(f'label shape: {lbls.shape}')
        print(f'ROI: {lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]):.4f}')
        print(f'BG: {1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]):.4f}')
        if i > 0:
            break