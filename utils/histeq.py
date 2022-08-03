import numpy as np
from PIL import Image, ImageOps

class HistEqualization(object):
    """Histogram Equalization
    
    Args:
        ...
    """

    def __init__(self, lbl=None):
        self.mask = lbl

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image):
            lbl (PIL Image):
        Returns:
            PIL Image:
            PIL Image:
        """
        return ImageOps.equalize(img, mask=self.mask), lbl

    def __repr__(self):
        return self.__class__.__name__ + '()'

if __name__ == "__main__":

    import os
    from tqdm import tqdm
    from PIL import Image, ImageOps
    from matplotlib import pyplot as plt

    split_f = '/data1/sdi/datasets/CPN_six/splits/train.txt'
    image_dir = '/data1/sdi/datasets/CPN_all/Images'
    mask_dir = '/data1/sdi/datasets/CPN_all/Masks'

    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]
    
    images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
    masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]

    for i in tqdm(range(len(images))):
        img = Image.open(images[i]).convert('L')
        ma = Image.open(masks[i]).convert('L')
        result = ImageOps.equalize(img, ma)
        result_HE = ImageOps.equalize(img, )
        result_HE.save('/data1/sdi/datasets/CPN_all_HE/Images/{}'.format(os.path.basename(images[i])))
        result.save('/data1/sdi/datasets/CPN_all_rHE/Images/{}'.format(os.path.basename(images[i])))
        plt.subplot(141), plt.imshow(img, cmap='gray'), plt.title('Image')
        plt.subplot(142), plt.imshow(np.array(result), cmap='gray'), plt.title('rHE'), plt.axis('off')
        plt.subplot(143), plt.imshow(np.array(result_HE), cmap='gray'), plt.title('HE'), plt.axis('off')
        plt.subplot(144), plt.imshow(ma, cmap='gray'), plt.title('Mask'), plt.axis('off')
        plt.savefig('/data1/sdi/datasets/HE/{}'.format(os.path.basename(images[i]).split('.')[0]))
        plt.close()
        plt.hist(np.array(img, dtype=np.uint8)[np.where(np.array(ma) > 0)], bins=[i for i in range(255)], density=True, align='mid')
        plt.title('PMF')
        plt.xlabel('pixel intensity')
        plt.ylabel('density')
        plt.savefig('/data1/sdi/datasets/hist/{}'.format(os.path.basename(images[i]).split('.')[0]))
        break
    