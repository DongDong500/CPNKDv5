import datasets as dt
from utils import ext_transforms as et


def get_dataset(opts, dataset, dver):
    
    mean = [0.485, 0.456, 0.406] if (opts.in_channels == 3) else [0.485]
    std = [0.229, 0.224, 0.225] if (opts.in_channels == 3) else [0.229]

    if opts.is_gaussian_crop:
        train_transform = et.ExtCompose([
            et.ExtResize(size=opts.resize, is_resize=opts.is_resize),
            et.ExtGaussianRandomCrop(size=opts.crop_size, 
                                        normal_h=opts.gaussian_crop_H, 
                                        normal_w=opts.gaussian_crop_W,
                                        block_size=opts.gaussian_crop_block_size),
            et.ExtScale(scale=opts.scale_factor, is_scale=opts.is_scale),
            et.ExtRandomVerticalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            ])
        val_transform = et.ExtCompose([
            et.ExtResize(size=opts.resize_val, is_resize=opts.is_resize_val),
            et.ExtGaussianRandomCrop(size=opts.crop_size_val, 
                                        normal_h=opts.gaussian_crop_H, 
                                        normal_w=opts.gaussian_crop_W,
                                        block_size=opts.gaussian_crop_block_size),
            et.ExtScale(scale=opts.scale_factor_val, is_scale=opts.is_scale_val),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            ])
        test_transform = et.ExtCompose([
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
        train_transform = et.ExtCompose([
            et.ExtResize(size=opts.resize, is_resize=opts.is_resize),
            et.ExtRandomCrop(size=opts.crop_size, is_crop=opts.is_crop, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor, is_scale=opts.is_scale),
            et.ExtRandomVerticalFlip(),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            et.GaussianPerturb(mean=opts.mu, std=opts.std)
            ])
        val_transform = et.ExtCompose([
            et.ExtResize(size=opts.resize_val, is_resize=opts.is_resize_val),
            et.ExtRandomCrop(size=opts.crop_size_val, is_crop=opts.is_crop_val, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor_val, is_scale=opts.is_scale_val),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            et.GaussianPerturb(mean=opts.mu_val, std=opts.std_val)
            ])
        test_transform = et.ExtCompose([
            et.ExtResize(size=opts.resize_test, is_resize=opts.is_resize_test),
            et.ExtRandomCrop(size=opts.crop_size_test, is_crop=opts.is_crop_test, pad_if_needed=True),
            et.ExtScale(scale=opts.scale_factor_test, is_scale=opts.is_scale_test),
            et.ExtToTensor(),
            et.ExtNormalize(mean=mean, std=std),
            et.GaussianPerturb(mean=opts.mu_test, std=opts.std_test)
            ])

    train_dst = dt.getdata.__dict__[dataset](root=opts.data_root, 
                                                    datatype=dataset, 
                                                    dver=dver, 
                                                    image_set='train', 
                                                    transform=train_transform, 
                                                    is_rgb=(opts.in_channels == 3), 
                                                    tvs=opts.tvs,
                                                    mu=opts.c_mu,
                                                    std=opts.c_std,
                                                    ratio=opts.pseudo_lbl_ratio)

    val_dst = dt.getdata.__dict__[dataset](root=opts.data_root, 
                                                    datatype=dataset, 
                                                    dver=dver, 
                                                    image_set='val', 
                                                    transform=val_transform, 
                                                    is_rgb=(opts.in_channels == 3), 
                                                    tvs=opts.tvs,
                                                    mu=opts.c_mu,
                                                    std=opts.c_std,
                                                    ratio=opts.pseudo_lbl_ratio)

    test_dst = dt.getdata.__dict__[dataset](root=opts.data_root, 
                                                    datatype=dataset, 
                                                    dver=dver, 
                                                    image_set='test', 
                                                    transform=test_transform, 
                                                    is_rgb=(opts.in_channels == 3))

    print("Dataset: %s\n\tTrain\t%d\n\tVal\t%d\n\tTest\t%d" % 
            (dver + '/' + dataset, len(train_dst), len(val_dst), len(test_dst)))

    return train_dst, val_dst, test_dst


if __name__ == "__main__":
    '''
        test code
    '''